# Visualize results

# general packages
import argparse
import numpy as np
import random
import os
import errno
import time
import math
import cv2
from collections import defaultdict

# torch
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
import torch.distributed as dist
import torch.utils.data.distributed
import torchvision.transforms as transforms

# util
from data.yc2_test_dataset import Yc2TestDataset, yc2_test_collate_fn
from model.dvsa import DVSA
from tools.test_util import compute_ba, print_results

parser = argparse.ArgumentParser()

# Data input settings
parser.add_argument('--start_from', default='', help='path to a model checkpoint to initialize model weights from. Empty = dont')
parser.add_argument('--image_root', default='./data/yc2/video_segments_25fps')
parser.add_argument('--box_file', default='./data/yc2/annotations/yc2_bb_val_annotations.json')
parser.add_argument('--val_split', default=['validation'], type=str, nargs='+', help='validation data folder')
parser.add_argument('--dataset_file', default='./data/yc2/annotations/youcookii_annotations.json')
parser.add_argument('--num_workers', default=6, type=int)
parser.add_argument('--num_class', default=67, type=int)
parser.add_argument('--class_file', default='./data/class_file.csv', type=str)
parser.add_argument('--rpn_proposal_root', default='./data/yc2', type=str)
parser.add_argument('--roi_pooled_feat_root', default='./data/yc2/roi_pooled_feat', type=str)

# Model settings: General
parser.add_argument('--num_proposals', default=100, type=int)
parser.add_argument('--enc_size', default=128, type=int)
parser.add_argument('--accu_thresh', default=0.5, type=float)
parser.add_argument('--num_frm', default=5, type=int)

# Model settings: Object Interaction
parser.add_argument('--hidden_size', default=256, type=int)
parser.add_argument('--n_layers', default=1, type=int)
parser.add_argument('--n_heads', default=4, type=int)
parser.add_argument('--attn_drop', default=0.2, type=float, help='dropout for the object interaction transformer layer')

# Optimization: General
parser.add_argument('--valid_batch_size', default=1, type=int)
parser.add_argument('--vis_dropout', default=0.2, type=float, help='dropout for the visual embedding layer')

parser.add_argument('--seed', default=123, type=int, help='random number generator seed to use')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='use gpu')
parser.add_argument('--id', default='', type=str)

parser.set_defaults(cuda=False)
parser.set_defaults(vis_output=True)
args = parser.parse_args()

# arguments inspection
assert(args.valid_batch_size == 1)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed_all(args.seed)


def get_dataset(args):

    valid_dataset = Yc2TestDataset(args.class_file, args.dataset_file, args.val_split,\
                                   args.image_root, args.box_file, num_proposals=args.num_proposals, \
                                   rpn_proposal_root=args.rpn_proposal_root, \
                                   roi_pooled_feat_root=args.roi_pooled_feat_root, vis_output=args.vis_output)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.valid_batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              collate_fn=yc2_test_collate_fn)

    return valid_loader


def get_model(args):
    model = DVSA(args.num_class, enc_size=args.enc_size, dropout=args.vis_dropout, \
                 hidden_size=args.hidden_size, n_layers=args.n_layers, n_heads=args.n_heads, \
                 attn_drop=args.attn_drop, num_frm=args.num_frm)

    # Initialize the networks and the criterion
    if len(args.start_from) > 0:
        print("Initializing weights from {}".format(args.start_from))
        model.load_state_dict(torch.load(args.start_from, map_location=lambda storage, location: storage))

    # Ship the model to GPU, maybe
    if args.cuda:
            model = model.cuda()
    return model


def tensor2video(x_np, ind):
    T, H, W, C = x_np.shape
    if not os.path.isdir('./vis/'+args.id):
        os.mkdir('./vis/'+args.id)
    video = cv2.VideoWriter('./vis/'+args.id+'/'+str(ind)+'.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (W, H))

    for t in range(T):
        video.write(cv2.cvtColor(x_np[t,:,:,:], cv2.COLOR_RGB2BGR))

    video.release()


def main(args):

    print('loading dataset')
    valid_loader = get_dataset(args)

    print('building model')
    model = get_model(args)

    valid(model, valid_loader)


def valid(model, loader):
    model.eval() # evaluation mode

    ba_score = defaultdict(list) # box accuracy metric

    vid_ba_lst = []

    for iter, data in enumerate(loader):
        print('evaluating iter {}...'.format(iter))

        # box_batch: N x O x T/25 x 5 (id,ytl,xtl,ybr,xbr)
        # ytl=-1 if the object is outside/non-exist/occlusion
        (x_rpn_batch, obj_batch, box_batch, box_label_batch,
            img_notrans_batch, rpn_batch, rpn_original_batch, vis_name) = data
        x_rpn_batch = Variable(x_rpn_batch)
        obj_batch = Variable(obj_batch)
        rpn_batch = Variable(rpn_batch)

        if args.cuda:
            x_rpn_batch = x_rpn_batch.cuda()
            obj_batch = obj_batch.cuda()
            box_batch = box_batch.cuda()
            box_label_batch = box_label_batch.cuda()
            rpn_batch = rpn_batch.cuda() # N, num_frames, num_proposals, 4
            rpn_original_batch = rpn_original_batch.cuda() # w/o coordinate normalization

        # divide long segment into pieces
        attn_weights = model.output_attn(x_rpn_batch, obj_batch).data

        # qualitative results, generate attention mask
        # cuda out of memory, ship to cpu if necessary
        visualize_attn(img_notrans_batch, attn_weights, rpn_batch.data, box_batch,
                                   obj_batch.data, box_label_batch, vis_name, loader, args.vis_output)

        # quantitative results
        ba = compute_ba(attn_weights, rpn_original_batch, box_batch, obj_batch.data, box_label_batch, thresh=args.accu_thresh)

        hits, misses = 0, 0
        for (i,h,m) in ba:
            ba_score[i].append((h, m))
            hits+=h
            misses+=m

        if hits+misses != 0:
            vid_ba_lst.append((vis_name, hits*1./(hits+misses)))

    # save the ba score for each segment
    with open('ba-per-seg-'+args.id+'.txt', 'w') as f:
        for i in vid_ba_lst:
            f.write(','.join((i[0], str(i[1])))+'\n')


def visualize_attn(img_batch, attn_weights, rpn, box_batch, obj_batch, box_label_batch, vis_name, loader, vis_output=False):

        # img_batch has not been resized
        display_factor = 0.5
        bg_mask = 0.1

        N, C, T, H, W = img_batch.size()
        _, T_rp, num_proposals, _ = rpn.size()
        _, O, T_fm, num_proposals = attn_weights.size() # the size of feature map
        assert(T_fm == T_rp)

        attn_mask_output = []

        rpn = rpn.clone()
        rpn[:,:,:,0] = torch.floor(rpn[:,:,:,0]*W-0.5)
        rpn[:,:,:,2] = torch.ceil(rpn[:,:,:,2]*W-0.5)
        rpn[:,:,:,1] = torch.floor(rpn[:,:,:,1]*H-0.5)
        rpn[:,:,:,3] = torch.ceil(rpn[:,:,:,3]*H-0.5)
        rpn = rpn.int()

        attn_mask = img_batch.squeeze(0).permute(1, 2, 3, 0).contiguous().numpy()
        attn_mask = attn_mask[12::25]
        for i in range(O):
            # find object name
            class_dict = loader.dataset.class_dict
            class_lst = list(class_dict.keys())[list(class_dict.values()).index(obj_batch[0, i].item())]
            print(class_lst)

            for t in range(T_fm):
                frm_on_rpn = rpn[0, t]
                n = torch.max(attn_weights[0, i, t, :], dim=0)[1]
                h_range = [max(frm_on_rpn[n,1],0), max(frm_on_rpn[n,3],1)]
                w_range = [max(frm_on_rpn[n,0],0), max(frm_on_rpn[n,2],1)]

                # draw generated
                cv2.rectangle(attn_mask[t], (w_range[0], h_range[0]), (w_range[1], h_range[1]), (0, 1, 0), 2)
                cv2.putText(attn_mask[t], class_lst, (w_range[0], h_range[0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (1,1,1),
                                1)

            # draw the ground-truth bounding box
            matched_ind = torch.nonzero(box_label_batch[0]==obj_batch[0, i]).squeeze()
            if matched_ind.view(-1).size(0): # ndimension is incorrect for torch.tensor(1) and torch.Tensor()
                matched_box = torch.index_select(box_batch[0], 0, matched_ind)
                for t in range(matched_box.size(1)):
                    for o in range(matched_box.size(0)):
                        box_ins = matched_box[o, t, :]
                        if box_ins[0] != -1:
                            box_ins = (box_ins/2).long()

                            # draw gt
                            cv2.rectangle(attn_mask[t], (box_ins[2], box_ins[1]), (box_ins[4], box_ins[3]),
                                         (1, 0, 0), 2)
                            cv2.putText(attn_mask[t], class_lst, (box_ins[2], box_ins[1]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (1,1,1), 1)

        # write video file
        tensor2video((attn_mask*255.0).astype(np.uint8), vis_name)


if __name__ == "__main__":
    main(args)
