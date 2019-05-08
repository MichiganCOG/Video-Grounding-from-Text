
# general packages
import argparse
import numpy as np
import random
import os
import errno
import time
import math
from collections import defaultdict
import json

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
parser.add_argument('--box_file', default='./data/yc2/annotations/yc2_bb_val_annotations.json', help='annotation data used for evaluation, must match --val_split')
parser.add_argument('--val_split', default=['validation'], type=str, nargs='+', help='data split used for testing')
parser.add_argument('--num_workers', default=6, type=int)
parser.add_argument('--num_class', default=67, type=int)
parser.add_argument('--class_file', default='./data/class_file.csv', type=str)
parser.add_argument('--rpn_proposal_root', default='./data/yc2/roi_box', type=str)
parser.add_argument('--roi_pooled_feat_root', default='./data/yc2/roi_pooled_feat', type=str)

# Model settings: General
parser.add_argument('--num_proposals', default=20, type=int)
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

# Data submisison
parser.add_argument('--save_to', default='./submission_yc2_bb_val.json', help='Save predictions to this JSON file')

parser.set_defaults(cuda=False)
args = parser.parse_args()

# arguments inspection
assert(args.valid_batch_size == 1)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

test_mode = 'testing' in args.val_split

def get_dataset(args):
    valid_dataset = Yc2TestDataset(args.class_file, args.val_split,\
                                   None, args.box_file, num_proposals=args.num_proposals, \
                                   rpn_proposal_root=args.rpn_proposal_root, \
                                   roi_pooled_feat_root=args.roi_pooled_feat_root, \
                                   test_mode = test_mode)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.valid_batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              collate_fn=yc2_test_collate_fn)

    return valid_loader


def get_model(args):
    model = DVSA(args.num_class, enc_size=args.enc_size, dropout=args.vis_dropout, \
                 hidden_size=args.hidden_size, n_layers=args.n_layers, n_heads=args.n_heads, \
                 attn_drop=args.attn_drop, num_frm=args.num_frm, has_loss_weighting=True)

    # Initialize the networks and the criterion
    if len(args.start_from) > 0:
        print("Initializing weights from {}".format(args.start_from))
        checkpoint = torch.load(args.start_from,map_location=lambda storage, location: storage)
        model.load_state_dict(checkpoint)

    # Ship the model to GPU
    if args.cuda:
            model = model.cuda()
    return model


def main(args):
    print('loading dataset')
    valid_loader = get_dataset(args)

    print('building model')
    model = get_model(args)

    valid(model, valid_loader)


def valid(model, loader):
    model.eval() # evaluation mode

    ba_score = defaultdict(list) # box accuracy metric
    class_labels_dict = loader.dataset.get_class_labels_dict() #dictionary for class labels - indices to strings

    json_data = {}
    database = {}

    for iter, data in enumerate(loader):
        print('evaluating iter {}...'.format(iter))

        (x_rpn_batch, obj_batch, box_batch, box_label_batch,
                _, rpn_batch, rpn_original_batch, vis_name) = data

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

        # quantitative results
        ba, segment_dict = compute_ba(attn_weights, rpn_original_batch, box_batch, obj_batch.data, \
            box_label_batch, vis_name, thresh=args.accu_thresh, class_labels_dict=class_labels_dict)

        split, rec, video_name, segment = vis_name.split('_-_')

        if video_name not in database:
            database[video_name] = {}
            database[video_name]['recipe_type'] = rec
        if 'segments' not in database[video_name]:
            database[video_name]['segments'] = {}

        database[video_name]['segments'][int(segment)] = segment_dict

        for (i,h,m) in ba:
            ba_score[i].append((h, m))

    if not test_mode: #Annotations for the testing split are not publicly available
        print_results(ba_score)

    json_data['database'] = database 
    with open(args.save_to,'w') as f:
        json.dump(json_data,f)

    print('Submission file saved to: {}'.format(args.save_to))

if __name__ == "__main__":
    main(args)
