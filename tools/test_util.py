
# general packages
from collections import defaultdict

# torch
import torch
import torch.nn.functional as F
import numpy as np

from tools.recall_util import iou

def compute_ba(attn_weights, rpn_batch, box_batch, obj_batch, box_label_batch, vis_name, class_labels_dict, fps=1.0, thresh=0.5):
    # fps is the frame rate of the attention map
    # both rpn_batch and box_batch have fps=1
    _, T_rp, num_proposals, _ = rpn_batch.size()
    _, O, T_gt, _ = box_batch.size()
    T_attn = attn_weights.size(2)

    assert(T_rp == T_gt) # both sampled at 1fps
    print('# of frames in gt: {}, # of frames in resampled attn. map: {}'.format(T_gt, np.rint(T_attn/fps)))

    hits, misses = [0 for o in range(O)], [0 for o in range(O)]

    results = []
    pos_counter = 0
    neg_counter = 0
    segment_dict = {} #segment dictionary - to output results to JSON file
    all_objects = []
    for o in range(O):
        object_dict = {}
        if box_label_batch[0, o] not in obj_batch[0, :]:
            print('object {} is not grounded!'.format(box_label_batch[0, o]))
            continue # don't compute score if the object is not grounded
        obj_ind_in_attn = (obj_batch[0, :] == box_label_batch[0, o]).nonzero().squeeze()
        if obj_ind_in_attn.numel() > 1:
            obj_ind_in_attn = obj_ind_in_attn[0]
        else:
            obj_ind_in_attn = obj_ind_in_attn.item()

        new_attn_weights = attn_weights[0, obj_ind_in_attn]
        _, max_attn_ind = torch.max(new_attn_weights, dim=1)

        # uncomment this for the random baseline
        # max_attn_ind = torch.floor(torch.rand(T_attn)*num_proposals).long()
        label = class_labels_dict[box_label_batch[0,o].item()]
        object_dict = {'label':label}
        
        boxes = []
        for t in range(T_gt):
            if box_batch[0,o,t,0] == -1: # object is outside/non-exist/occlusion
                boxes.append({'xtl':-1, 'ytl':-1, 'xbr':-1, 'ybr':-1, 'outside':1, 'occluded':1}) #object is either occluded or outside of frame 
                neg_counter += 1
                continue
            pos_counter += 1
            box_ind = max_attn_ind[int(min(np.rint(t*fps), T_attn-1))]
            box_coord = rpn_batch[0, t, box_ind, :].view(1,4) # x_tl, y_tl, x_br, y_br
            gt_box = box_batch[0,o,t][torch.Tensor([2,1,4,3]).type(box_batch.type()).long()].view(1,4) # inverse x and y

            if iou(box_coord, gt_box)[0,0] > thresh:
                hits[o] += 1
            else:
                misses[o] += 1

            xtl = box_coord[0][0].item()
            ytl = box_coord[0][1].item()
            xbr = box_coord[0][2].item()
            ybr = box_coord[0][3].item()
            boxes.append({'xtl':xtl, 'ytl':ytl, 'xbr':xbr, 'ybr':ybr, 'outside':0, 'occluded':0}) 

        object_dict['boxes'] = boxes
        all_objects.append(object_dict)

        results.append((box_label_batch[0, o].item(), hits[o], misses[o]))
    segment_dict['objects'] = all_objects

    print('percentage of frames with box: {}'.format(pos_counter/(pos_counter+neg_counter)))
    return results, segment_dict

def print_results(ba_score):
    print('-'*80)
    print('-'*30+' Evaluation results '+'-'*30)
    print('-'*80)

    ba_final = []
    for k, r in ba_score.items():
        cur_hit = 0
        cur_miss = 0
        for v in r:
            cur_hit += v[0]
            cur_miss += v[1]

        if cur_hit+cur_miss != 0:
            print('BA for {}(...): {:.4f}'.format(k, cur_hit/(cur_hit+cur_miss)))
            ba_final.append(cur_hit/(cur_hit+cur_miss))

    print('The overall BA is: {:.4f}'.format(np.mean(ba_final)))
