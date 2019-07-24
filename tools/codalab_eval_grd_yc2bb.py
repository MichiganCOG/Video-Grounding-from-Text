#Adapted from: https://github.com/facebookresearch/ActivityNet-Entities/blob/master/scripts/eval_grd_anet_entities.py  

#Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Evaluation script for object grounding over generated sentences

import sys
import os
os.system('pip3 install torch==0.4.0')
os.system('pip3 install numpy')
os.system('pip3 install tqdm')

import json
import csv
import argparse
import torch
import itertools
import numpy as np
from collections import defaultdict
from recall_util import iou

from tqdm import tqdm

class YC2BBGrdEval(object):

    def __init__(self, reference_file=None, submission_file=None, class_file=None,
                 iou_thresh=0.5, verbose=False):

        if not reference_file:
            raise IOError('Please input a valid reference file!')
        if not submission_file:
            raise IOError('Please input a valid submission file!')

        self.iou_thresh = iou_thresh
        self.verbose = verbose

        self.import_ref(reference_file)
        self.import_sub(submission_file)
        self.class_dict = get_class_labels(class_file)


    def import_ref(self, reference_file=None):

        with open(reference_file) as f:
            ref = json.load(f)['database']

        self.ref = ref

    def import_sub(self, submission_file=None):

        with open(submission_file) as f:
            pred = json.load(f)['database']
        self.pred = pred

    #sort string list of number by natural sorting
    def sort_keys(self, keys):
        temp = sorted([int(k) for k in keys]) #sorted as integers

        return [str(k) for k in temp] #sorted as string

    #get annotation for all objects
    def get_all_bboxes(self, ann):
        ref_bbox_all = []

        for v in ann['objects']:
            ref_bbox_all.append(self.get_object_bboxes(v))

        ref_bbox_all = torch.stack(ref_bbox_all)
        if ref_bbox_all.dim() == 2: #If there's only one label, unsqueeze the first dimension
            ref_bbox_all.unsqueeze_(0)

        return ref_bbox_all

    #get annotation for a single object
    def get_object_bboxes(self, ann):
        ref_bbox = torch.Tensor([])

        for v in ann['boxes']:
            #Do not include boxes that are occluded/outside of screen
            if v['outside'] or v['occluded']:
                bbox = torch.Tensor([-1,-1,-1,-1])
            else:
                bbox = torch.Tensor([v['xtl'], v['ytl'], v['xbr'], v['ybr']])
            ref_bbox = torch.cat((ref_bbox, bbox.unsqueeze(0)))
        return ref_bbox

    def gt_grd_eval(self):
        class_labels_dict = get_class_labels_dict(self.class_dict)

        ref = self.ref
        pred = self.pred
        print('Number of videos in the reference: {}, number of videos in the submission: {}'.format(len(ref), len(pred)))

        results = defaultdict(list)
        for vid, anns in ref.items():
            for seg, ann in anns['segments'].items():
                if len(ann) == 0:
                    continue # annotation not available

                ref_bbox_all = self.get_all_bboxes(ann) #return all bboxes as 3-D tensor

                for obj in range(len(ann['objects'])):
                    ref_bbox = ref_bbox_all[int(obj)] # select matched boxes
                    valid_idx = (1 - (ref_bbox == -1))[:,0]

                    # Note that despite discouraged, a single word could be annotated across multiple boxes/frames
                    assert(ref_bbox.size(0) > 0)

                    class_name = class_labels_dict[self.class_dict[ann['objects'][obj]['label']]] #class name (invariant to plural forms)

                    if vid not in pred:
                        print('{} not grounded'.format(vid)) # video not grounded
                        continue 
                    elif seg not in pred[vid]['segments']:
                        print('{} not grounded'.format(seg)) # segment not grounded
                        continue 
                    elif obj > len(ann['objects']):
                        print('{} not grounded'.format(class_name)) # object not grounded
                        continue
                    else:
                        pred_obj = obj  
                        pred_bbox = self.get_object_bboxes(pred[vid]['segments'][seg]['objects'][pred_obj])
                        
                        overlap = torch.diag(iou(pred_bbox, ref_bbox))
                        overlap = overlap[valid_idx] #ignore un-annotated entries

                        hits = int(torch.sum(overlap > self.iou_thresh))
                        misses = len(overlap) - hits
                        results[class_name].append((hits, misses))

        print('Number of groundable objects in this split: {}'.format(len(results)))
        accu_per_clss = {}
        grd_accu = []
        for i, val in results.items():
            cur_hits = 0
            cur_miss = 0

            for hm in val:
                cur_hits += hm[0]
                cur_miss += hm[1]

            if cur_hits+cur_miss == 0:
                continue

            acc = cur_hits/(cur_hits+cur_miss)
            grd_accu.append(acc)
            accu_per_clss[i] = acc

        grd_accu = np.mean(grd_accu)

        if self.verbose:
            print('Grounding accuracy per class:')
            for idx in range(len(results)):
                label = class_labels_dict[idx]
                if label not in accu_per_clss: #object not grounded
                    continue 

                acc = accu_per_clss[label] 
                print('{}: {:.4f}'.format(label, acc))

        print('-' * 80)
        print('The overall localization accuracy is {:.4f}'.format(grd_accu))
        print('-' * 80)
        return grd_accu

def get_class_labels(class_file):
    class_dict = {} # both singular form & plural form are associated with the same label
    with open(class_file) as f:
        cls = csv.reader(f, delimiter=',')
        for i, row in enumerate(cls):
            for r in range(1, len(row)):
                if row[r]:
                    class_dict[row[r]] = int(row[0])

    return class_dict

# return an inverse mapping - indices to labels
def get_class_labels_dict(class_dict):
    return {v:k for k,v in class_dict.items()}

def main():

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    submit_file = os.path.join(input_dir, 'res', 'submission_yc2_bb.json')
    ref_file = os.path.join(input_dir, 'ref', 'reference_yc2_bb.json')
    class_file = os.path.join(input_dir, 'ref','class_file.csv')

    grd_evaluator = YC2BBGrdEval(reference_file=ref_file, submission_file=submit_file, class_file=class_file,
                           iou_thresh=0.5, verbose=False)

    grd_accu = grd_evaluator.gt_grd_eval()

    output_filename = os.path.join(output_dir, 'scores.txt')
    with open(output_filename, 'w') as f:
        f.write('box_accuracy: {:.4f}'.format(grd_accu))

if __name__=='__main__':
    main()
