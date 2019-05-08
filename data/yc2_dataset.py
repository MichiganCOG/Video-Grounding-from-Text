
import json
import csv
import os
import scipy.io
import numpy as np
from PIL import Image
import math

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
    split = split_lst[0] # split_lst contains only 'training' or 'validation'

    with open(gt_box_file, 'r') as data_file:
        data_all = json.load(data_file)
    data = data_all['database']

    for vid, val in data.items():
        segments = val['segments']
        rec = val['recipe_type']
        for ann, objects in segments.items():

            if 'sentence' in objects: # for now, training json file only contains full sentence
                segment_labels = objects['sentence']
            else: # validation json file is completely annotated per object
                segment_labels = []
                for obj in objects['objects']:
                    segment_labels.append(obj['label'])
            split_sentences.append(segment_labels)
            split_segments.append((split, rec, vid, str(ann).zfill(2))) # tuple of id (split, rec, vid, seg)

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


class Yc2Dataset(Dataset):
    def __init__(self, class_file, split_lst, vid_data_file,\
                 num_proposals, rpn_proposal_root, roi_pooled_feat_root, \
                 num_class=67, num_frm=5):
        super().__init__()

        # read in the object list as a dict {class_name=class_id}
        sentences_proc, segments_tuple = get_segments_and_sentences(vid_data_file, split_lst)
        self.class_dict = get_class_labels(class_file)
        assert(num_class == len(set(self.class_dict.values())))
        self.num_class = num_class
        self.num_frm = num_frm
        self.vid_data_file = vid_data_file 

        assert(len(sentences_proc) == len(segments_tuple))

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

        self.sample_lst = [] # list of id tuple, object labels

        # read data which contains video dimensions, training data contains no bounding box annotations
        with open(self.vid_data_file, 'r') as f:
            self.data_all = json.load(f)

        for i, t in enumerate(segments_tuple):
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
        # sample positive sample
        sample = self.sample_lst[index]
        split = sample[0][0]
        rec = sample[0][1]
        vid = sample[0][2]
        seg = sample[0][3]
        max_num_obj = 15
        num_frm = self.num_frm

        # roi region feature
        x_rpn = []
        frm=1

        feat_name = vid+'_'+seg+'.pth'
        img_name = vid+'_'+seg+'_'+str(frm).zfill(4)+'.jpg'
        x_rpn = torch.load(os.path.join(self.roi_pooled_feat_root, split, feat_name))
        while self.rpn_dict.get(img_name, -1) > -1:
            ind = self.rpn_dict[img_name]
            frm+=1
            img_name = vid+'_'+seg+'_'+str(frm).zfill(4)+'.jpg'

        x_rpn = x_rpn.permute(2,0,1).contiguous() # encoding size x number of frames x number of proposals
        x_rpn = x_rpn[:, :, :self.num_proposals]

        # normalize coordidates to 0-1
        rwidth = self.data_all['database'][vid]['rwidth']
        rheight = self.data_all['database'][vid]['rheight']

        # random sample 5 frames from 5 uniform intervals
        T = x_rpn.size(1)
        itv = T*1./num_frm
        ind = [min(T-1, int((i+np.random.rand())*itv)) for i in range(num_frm)]
        x_rpn = x_rpn[:, ind, :]

        obj_tensor = torch.LongTensor(sample[1])
        obj_tensor = torch.cat((obj_tensor, torch.LongTensor(max_num_obj-len(sample[1])).fill_(self.num_class))) # padding
        pos_sample = [x_rpn, obj_tensor]

        # sample negative sample
        total_s = len(self.sample_lst)
        neg_index = np.random.randint(total_s)
        while len(set(sample[1]).intersection(set(self.sample_lst[neg_index][1]))) != 0: # shouldn't have any overlapping object
            neg_index = np.random.randint(total_s)

        sample = self.sample_lst[neg_index]
        split = sample[0][0]
        rec = sample[0][1]
        vid = sample[0][2]
        seg = sample[0][3]

        # roi region feature
        x_rpn = []
        frm=1

        feat_name = vid+'_'+seg+'.pth'
        img_name = vid+'_'+seg+'_'+str(frm).zfill(4)+'.jpg'
        x_rpn = torch.load(os.path.join(self.roi_pooled_feat_root, split, feat_name))
        while self.rpn_dict.get(img_name, -1) > -1:
            ind = self.rpn_dict[img_name]

            frm+=1
            img_name = vid+'_'+seg+'_'+str(frm).zfill(4)+'.jpg'

        x_rpn = x_rpn.permute(2,0,1).contiguous() # encoding size x number of frames x number of proposals
        x_rpn = x_rpn[:, :, :self.num_proposals]

        # normalize coordidates to 0-1
        rwidth = self.data_all['database'][vid]['rwidth']
        rheight = self.data_all['database'][vid]['rheight']

        # random sample 5 frames from 5 uniform intervals
        T = x_rpn.size(1)
        itv = T*1./num_frm
        ind = [min(T-1, int((i+np.random.rand())*itv)) for i in range(num_frm)]
        x_rpn = x_rpn[:, ind, :]

        obj_tensor = torch.LongTensor(sample[1])
        obj_tensor = torch.cat((obj_tensor, torch.LongTensor(max_num_obj-len(sample[1])).fill_(self.num_class))) # pad -1
        neg_sample = [x_rpn, obj_tensor]

        return [torch.stack(i) for i in zip(pos_sample, neg_sample)]


def yc2_collate_fn(batch_lst):
        batch_size = len(batch_lst)
        assert(batch_size == 1)

        x_rpn_batch = batch_lst[0][0]
        obj_batch = torch.LongTensor(batch_lst[0][1])

        return (x_rpn_batch, obj_batch)
