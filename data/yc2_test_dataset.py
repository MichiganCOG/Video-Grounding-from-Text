
import json
import csv
import os
import scipy.io
import numpy as np
from PIL import Image

import torch
import torchtext
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def get_segments_and_sentences(gt_box_file, split_lst):
    # build vocab and tokenized sentences
    text_proc = torchtext.data.Field(sequential=True, tokenize='spacy',
                                lower=True, batch_first=True)
    split_sentences = []
    split_segments = []
    split = split_lst[0] #split_lst contains only 'validation' or 'testing'

    with open(gt_box_file, 'r') as data_file:
        data_all = json.load(data_file)
    data = data_all['database']
    
    for vid, val in data.items():
        segments = val['segments']
        rec = val['recipe_type']
        for ann, objects in segments.items():
            segment_labels = []
            for obj in objects['objects']:
                segment_labels.append(obj['label'])
            split_sentences.append(segment_labels)
            split_segments.append((split, rec, vid, str(ann).zfill(2))) #tuple of id (split, vid, seg)

    sentences_proc = list(map(text_proc.preprocess, split_sentences)) # build vocab on train and val

    print('# of sentences in this split: {}'.format(len(sentences_proc)))

    return sentences_proc, split_segments


def get_class_labels(class_file):
    class_dict = {} # both singular form & plural form are associated with the same label
    with open(class_file) as f:
        cls = csv.reader(f, delimiter=',')
        for i, row in enumerate(cls):
            for r in range(1, len(row)):
                if row[r]:
                    class_dict[row[r]] = int(row[0])

    return class_dict


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class Yc2TestDataset(Dataset):
    def __init__(self, class_file, split_lst, image_root, gt_box_file, \
                 num_proposals, rpn_proposal_root, roi_pooled_feat_root, test_mode=False, vis_output=False):
        super().__init__()

        # read in the object list as a dict {class_name=class_id}
        sentences_proc, segments_tuple = get_segments_and_sentences(gt_box_file, split_lst)
        self.class_dict = get_class_labels(class_file)
        self.image_root = image_root
        self.gt_box_file = gt_box_file
        self.loader = pil_loader
        self.test_mode = test_mode
        self.vis_output = vis_output

        self.spatial_transform_notrans = transforms.Compose([
            transforms.ToTensor()
            ])

        assert(len(sentences_proc) == len(segments_tuple))

        if self.test_mode:
            print('*'*62)
            print('*  [WARNING] Eval unavailable for the test set!  *\
                \n* Please submit your results to the eval server!  *')
            print('*'*62)

        # read rpn object proposals
        self.rpn_dict = {}
        self.rpn_chunk = []
        for s_ind, s in enumerate(split_lst):
            total_num_proposals = 100 # always load all the proposals we have
            rpn_lst_file = os.path.join(rpn_proposal_root, s+'-box-'+str(total_num_proposals)+'.txt')
            rpn_chunk_file = os.path.join(rpn_proposal_root, s+'-box-'+str(total_num_proposals)+'.pth')
            key_counter = len(self.rpn_dict)
            with open(rpn_lst_file) as f:
                rpn_lst = f.readline().split(',')
                self.rpn_dict.update({r.strip():(i+key_counter) for i,r in enumerate(rpn_lst)})

            self.rpn_chunk.append(torch.load(rpn_chunk_file))
        self.rpn_chunk = torch.cat(self.rpn_chunk).cpu()
        assert(self.rpn_chunk.size(0) == len(self.rpn_dict))
        assert(self.rpn_chunk.size(2) == 4)

        self.num_proposals = num_proposals
        self.roi_pooled_feat_root = roi_pooled_feat_root

        with open(self.gt_box_file, 'r') as f:
            self.data_all = json.load(f)

        self.sample_lst = [] # list of id tuple, object labels
        
        # read gt bounding boxes O x T/25 x (id, ytl, xtl, ybr, xbr)
        # coordinates are 0-indexed
        for i, t in enumerate(segments_tuple):
            vid = t[2]
            seg = str(int(t[3]))

            # if video has no annotations, continue
            if not vid in self.data_all['database']:
                continue

            # check if ground truth bounding box exists for segment
            if seg in self.data_all['database'][vid]['segments'].keys():
                s = sentences_proc[i]
                inc_flag = 0
                obj_label = []
                for w in s:
                    if self.class_dict.get(w, -1) >= 0:
                        obj_label.append(self.class_dict[w])
                        inc_flag = 1

                if inc_flag:
                    self.sample_lst.append((t, obj_label))

        print('# of segments for {}: {}, percentage in the raw data: {:.2f}'.format(
               split_lst, len(self.sample_lst), len(self.sample_lst)/len(sentences_proc)))


    def __len__(self):
        return len(self.sample_lst)


    def __getitem__(self, index):
        sample = self.sample_lst[index]
        split = sample[0][0]
        rec = sample[0][1]
        vid = sample[0][2]
        seg = sample[0][3]

        gt_box = self.data_all['database'][vid]['segments'][str(int(seg))]['objects']
        rwidth = self.data_all['database'][vid]['rwidth']
        rheight = self.data_all['database'][vid]['rheight']
        
        assert(len(gt_box)>0) # non-empty

        num_frames_1fps = len(gt_box[0]['boxes']) #frames are annotated at 1fps
        box = torch.Tensor(len(gt_box), num_frames_1fps, 5).fill_(-1.0) # -1 if no object appears
        box_label = torch.LongTensor(len(gt_box))
        for i, b in enumerate(gt_box):
            cls_label = self.class_dict[b['label']] 
            box_label[int(i)] = cls_label
            for j, v in enumerate(b['boxes']):
                if self.test_mode: #Annotations for the testing split are not publicly available
                    box[int(i), int(j)] = torch.Tensor([cls_label, -1, -1, -1, -1])
                elif v['occluded'] == 0 and v['outside'] == 0:
                    box[int(i), int(j)] = torch.Tensor([cls_label, v['ytl'], v['xtl'], v['ybr'], v['xbr']])

        num_frames = num_frames_1fps * 25 #video sampled at 25fps
        if self.vis_output:
            image_path = os.path.join(self.image_root, split, rec, vid, seg)
            img_notrans = []
            for i in range(num_frames):
                img_notrans.append(self.spatial_transform_notrans(self.loader(os.path.join(image_path, '{:04d}.jpg'.format(i+1)))))
            img_notrans = torch.stack(img_notrans, dim=1) # 3, T, H, W
        else:
            # no need to load raw images
            img_notrans = torch.zeros(3, num_frames, 1, 1) # dummy

        # rpn object propoals
        rpn = []
        x_rpn = []
        frm=1

        feat_name = vid+'_'+seg+'.pth'
        img_name = vid+'_'+seg+'_'+str(frm).zfill(4)+'.jpg'
        x_rpn = torch.load(os.path.join(self.roi_pooled_feat_root, split, feat_name))
        while self.rpn_dict.get(img_name, -1) > -1:
            ind = self.rpn_dict[img_name]
            rpn.append(self.rpn_chunk[ind])
            frm+=1
            img_name = vid+'_'+seg+'_'+str(frm).zfill(4)+'.jpg'

        rpn = torch.stack(rpn) # number of frames x number of proposals per frame x 4
        rpn = rpn[:, :self.num_proposals, :]

        x_rpn = x_rpn.permute(2,0,1).contiguous() # encoding size x number of frames x number of proposals
        x_rpn = x_rpn[:, :, :self.num_proposals]

        rpn_original = rpn-1 # convert to 1-indexed

        # normalize coordidates to 0-1
        # coordinates are 1-indexed:  (x_tl, y_tl, x_br, y_br)
        rpn[:, :, 0] = (rpn[:, :, 0]-0.5)/rwidth
        rpn[:, :, 2] = (rpn[:, :, 2]-0.5)/rwidth
        rpn[:, :, 1] = (rpn[:, :, 1]-0.5)/rheight
        rpn[:, :, 3] = (rpn[:, :, 3]-0.5)/rheight

        assert(torch.max(rpn) <= 1)

        vis_name = '_-_'.join((split, rec, vid, seg))

        return (x_rpn, sample[1], box, box_label, img_notrans, rpn, rpn_original, vis_name)
    
    # return an inverse mapping - indices to labels
    def get_class_labels_dict(self):
        return {v:k for k,v in self.class_dict.items()}


def yc2_test_collate_fn(batch_lst):
        batch_size = len(batch_lst)
        assert(batch_size == 1)

        x_rpn_batch = batch_lst[0][0].unsqueeze(0)
        obj_batch = torch.LongTensor(batch_lst[0][1]).unsqueeze(0)
        box_batch = batch_lst[0][2].unsqueeze(0)
        box_label_batch = batch_lst[0][3].unsqueeze(0)
        img_notrans_batch = batch_lst[0][4].unsqueeze(0)
        rpn_batch = batch_lst[0][5].unsqueeze(0)
        rpn_original_batch = batch_lst[0][6].unsqueeze(0)
        vis_batch = batch_lst[0][7]

        return (x_rpn_batch, obj_batch, box_batch, box_label_batch, img_notrans_batch, rpn_batch, rpn_original_batch, vis_batch)
