
# general packages
import argparse
import numpy as np
import random
import os
import errno
import time

# torch
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import torch.utils.data.distributed

# util
from data.yc2_dataset import Yc2Dataset, yc2_collate_fn
from model.dvsa import DVSA

parser = argparse.ArgumentParser()

# Data input settings
parser.add_argument('--start_from', default='', help='path to a model checkpoint to initialize model weights from. Empty = dont')
parser.add_argument('--vid_data_file', default='./data/yc2/annotations/yc2_training_vid.json', help='contains original video coordinates, used in normalization step')
parser.add_argument('--val_box_file', default='./data/yc2/annotations/yc2_bb_val_annotations.json', help='only used to read video dimensions, bounding boxes are not used')
parser.add_argument('--train_split', default=['training'], type=str, nargs='+', help='training data folder')
parser.add_argument('--val_split', default=['validation'], type=str, nargs='+',  help='validation data folder')
parser.add_argument('--num_workers', default=6, type=int)
parser.add_argument('--rpn_proposal_root', default='./data/yc2/roi_box', type=str)
parser.add_argument('--roi_pooled_feat_root', default='./data/yc2/roi_pooled_feat', type=str)

# Model settings: General
parser.add_argument('--loss_weighting',help='perform loss weighting on each frame during training',action='store_true')
parser.add_argument('--loss_factor', default=0.9, type=float)
parser.add_argument('--obj_interact', help='add object interaction to model during training',action='store_true') 
parser.add_argument('--num_class', default=67, type=int)
parser.add_argument('--class_file', default='./data/class_file.csv', type=str)
parser.add_argument('--num_proposals', default=20, type=int)
parser.add_argument('--enc_size', default=128, type=int)
parser.add_argument('--ranking_margin', default=0.1, type=float)

# Model settings: Object Interaction
parser.add_argument('--hidden_size', default=256, type=int)
parser.add_argument('--n_layers', default=1, type=int)
parser.add_argument('--n_heads', default=4, type=int)
parser.add_argument('--attn_drop', default=0.2, type=float, help='dropout for the object interaction transformer layer')

# Optimization: General
parser.add_argument('--max_epochs', default=30, type=int, help='max number of epochs to run for')
parser.add_argument('--batch_size', default=1, type=int, help='what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
parser.add_argument('--valid_batch_size', default=1, type=int)
parser.add_argument('--vis_dropout', default=0.2, type=float, help='dropout for the visual embedding layer')
parser.add_argument('--num_frm', default=5, type=int)

# Optimization
parser.add_argument('--optim',default='sgd', help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
parser.add_argument('--learning_rate', default=0.05, type=float, help='learning rate')
parser.add_argument('--alpha', default=0.9, type=float, help='alpha for adagrad/rmsprop/momentum/adam')
parser.add_argument('--beta', default=0.999, type=float, help='beta used for adam')
parser.add_argument('--epsilon', default=1e-8, help='epsilon that goes into denominator for smoothing')
parser.add_argument('--loss_alpha_r', default=2, type=int, help='The weight for regression loss')
parser.add_argument('--patience_epoch', default=1, type=int, help='Epoch to wait to determine a plateau')
parser.add_argument('--reduce_factor', default=0.5, type=float, help='Factor of learning rate reduction')
parser.add_argument('--grad_norm', default=1, type=float, help='Gradient clipping norm')

# Data parallel
parser.add_argument('--dist_url', default='file://'+os.getcwd()+'/nonexistent_file', type=str, help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')

# Evaluation/Checkpointing
parser.add_argument('--save_checkpoint_every', default=1, type=int, help='how many epochs to save a model checkpoint?')
parser.add_argument('--checkpoint_path', default='./checkpoint', help='folder to save checkpoints into (empty = this folder)')
parser.add_argument('--losses_log_every', default=1, type=int, help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

parser.add_argument('--seed', default=123, type=int, help='random number generator seed to use')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='use gpu')
parser.add_argument('--enable_visdom', action='store_true', help='enable output to visdom server')

parser.set_defaults(cuda=False)
parser.set_defaults(loss_weighting=False)
args = parser.parse_args()

# arguments inspection
assert(args.batch_size == 1)
assert(args.valid_batch_size == 1)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed_all(args.seed)


def get_dataset(args):
    train_dataset = Yc2Dataset(args.class_file, args.train_split, args.vid_data_file, \
                               num_proposals=args.num_proposals, rpn_proposal_root=args.rpn_proposal_root,\
                               roi_pooled_feat_root=args.roi_pooled_feat_root, num_class=args.num_class, \
                               num_frm=args.num_frm)

    args.distributed = args.world_size > 1
    if args.distributed and args.cuda:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=args.num_workers,
                              collate_fn=yc2_collate_fn)

    valid_dataset = Yc2Dataset(args.class_file, args.val_split, args.val_box_file, \
                               num_proposals=args.num_proposals, rpn_proposal_root=args.rpn_proposal_root,\
                               roi_pooled_feat_root=args.roi_pooled_feat_root, num_class=args.num_class, \
                               num_frm=args.num_frm)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.valid_batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              collate_fn=yc2_collate_fn)

    return train_loader, train_sampler, valid_loader


def get_model(args):
    model = DVSA(args.num_class, enc_size=args.enc_size, dropout=args.vis_dropout, \
                hidden_size=args.hidden_size, n_layers=args.n_layers, n_heads=args.n_heads, \
                attn_drop=args.attn_drop, num_frm=args.num_frm, has_loss_weighting=args.loss_weighting)

    # Ship the model to GPU
    if args.cuda:
        if args.distributed:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            model = torch.nn.DataParallel(model).cuda()

    # Initialize the networks and the criterion
    if len(args.start_from) > 0:
        print("Initializing weights from {}".format(args.start_from))
        model.load_state_dict(torch.load(args.start_from, map_location=lambda storage,
            location: storage)['state_dict'], strict=False)

    return model


def main(args):
    try:
        os.makedirs(args.checkpoint_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise

    print('loading dataset')
    train_loader, train_sampler, valid_loader = get_dataset(args)

    print('building model')
    model = get_model(args)

    if args.optim == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            args.learning_rate, betas=(args.alpha, args.beta), eps=args.epsilon)
    elif args.optim == 'sgd': # original implementation in the paper
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            args.learning_rate,
            weight_decay=1e-4,
            momentum=args.alpha,
            nesterov=True
        )
    else:
        assert False, "only support adam or sgd"

    # learning rate decay
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.reduce_factor,
                                               patience=args.patience_epoch,
                                               verbose=True)

    best_loss = float('inf')
    
    if args.enable_visdom: 
        import visdom
        vis = visdom.Visdom(env='weakly-supervised')
        vis_window={'iter': None,
                'loss': None}

    all_cls_losses = []
    all_training_losses = []

    for train_epoch in range(args.max_epochs):
        t_epoch_start = time.time()
        print('Epoch: {}'.format(train_epoch))

        if args.distributed:
            train_sampler.set_epoch(train_epoch)

        epoch_loss = train(train_epoch, model, optimizer, train_loader, args, vis=None, vis_window=None)
        all_training_losses.append(epoch_loss)

        val_cls_loss = valid(model, valid_loader)
        all_cls_losses.append(val_cls_loss)

        # learning rate decay
        scheduler.step(val_cls_loss)

        if args.enable_visdom:
            if vis_window['loss'] is None:
                if not args.distributed or (args.distributed and dist.get_rank() == 0):
                    vis_window['loss'] = vis.line(
                    X=np.tile(np.arange(len(all_cls_losses)),
                              (2,1)).T,
                    Y=np.column_stack((np.asarray(all_training_losses),
                                       np.asarray(all_cls_losses))),
                    opts=dict(title='Loss',
                              xlabel='Validation Iter',
                              ylabel='Loss',
                              legend=['train',
                                      'dev_cls']))
            else:
                if not args.distributed or (
                    args.distributed and dist.get_rank() == 0):
                    vis.line(
                    X=np.tile(np.arange(len(all_cls_losses)),
                              (2, 1)).T,
                    Y=np.column_stack((np.asarray(all_training_losses),
                                       np.asarray(all_cls_losses))),
                    win=vis_window['loss'],
                    opts=dict(title='Loss',
                              xlabel='Validation Iter',
                              ylabel='Loss',
                              legend=['train',
                                      'dev_cls']))

        if val_cls_loss < best_loss:
            best_loss = val_cls_loss
            if (args.distributed and dist.get_rank() == 0) or not args.distributed:
                torch.save(model.module.state_dict(), os.path.join(args.checkpoint_path, 'model_best_loss.t7'))
            print('*'*5)
            print('Better validation loss {:.4f} found, save model'.format(val_cls_loss))

        # save eval and train losses
        if (args.distributed and dist.get_rank() == 0) or not args.distributed:
            torch.save({'train_loss':all_training_losses,
                        'eval_cls_loss':all_cls_losses,
                        }, os.path.join(args.checkpoint_path, 'model_losses.t7'))
        
        # validation/save checkpoint every few epochs
        if train_epoch%args.save_checkpoint_every == 0 or train_epoch == args.max_epochs:
            if (args.distributed and dist.get_rank() == 0) or not args.distributed:
                torch.save(model.module.state_dict(),
                   os.path.join(args.checkpoint_path, 'model_epoch_{}.t7'.format(train_epoch)))

        # all other process wait for the 1st process to finish
        if args.distributed:
            dist.barrier()

        print('-'*80)
        print('Epoch {} summary'.format(train_epoch))
        print('Train loss: {:.4f}, val loss: {:.4f}, Time: {:.4f}s'.format(
            epoch_loss, val_cls_loss, time.time()-t_epoch_start
        ))
        print('-'*80)


def train(epoch, model, optimizer, train_loader, args, vis=None, vis_window=None):
    model.train() # training mode
    train_loss = []
    nbatches = len(train_loader)
    t_iter_start = time.time()

    # import pdb; pdb.set_trace()

    for train_iter, data in enumerate(train_loader):
        (x_rpn_batch, obj_batch) = data

        x_rpn_batch = Variable(x_rpn_batch)
        obj_batch = Variable(obj_batch)

        if args.cuda:
            x_rpn_batch = x_rpn_batch.cuda()
            obj_batch = obj_batch.cuda()

        t_model_start = time.time()

        # N, C_out, num_class
        output, loss_weigh = model(x_rpn_batch, obj_batch)

        if args.loss_weighting or args.obj_interact: 
            rank_batch = F.margin_ranking_loss(output[:,0:1], output[:,1:2], Variable(
                torch.ones(output.size()).type(output.data.type())), margin=args.ranking_margin, reduce=False)
            if args.loss_weighting and args.obj_interact:
                loss_weigh = (output[:, 0:1]+loss_weigh)/2. # avg
            elif args.loss_weighting:
                loss_weigh = output[:,0:1]
            else:
                loss_weigh = loss_weigh.unsqueeze(1)
            # ranking loss
            cls_loss = args.loss_factor*(rank_batch*loss_weigh).mean()+ \
                        (1-args.loss_factor)*-torch.log(2*loss_weigh).mean()
        else:
            # ranking loss
            cls_loss = F.margin_ranking_loss(output[:,0:1], output[:,1:2], Variable(
                torch.Tensor([[1],[1]]).type(output.data.type())), margin=args.ranking_margin)

        optimizer.zero_grad()
        cls_loss.backward()

        # enable the clipping for zero mask loss training
        total_grad_norm = clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                         args.grad_norm)

        optimizer.step()

        train_loss.append(cls_loss.item())

        if args.enable_visdom:
            if vis_window['iter'] is None:
                if not args.distributed or (
                    args.distributed and dist.get_rank() == 0):
                    vis_window['iter'] = vis.line(
                        X=np.arange(epoch*nbatches+train_iter, epoch*nbatches+train_iter+1),
                        Y=np.asarray(train_loss),
                        opts=dict(title='Training Loss',
                                  xlabel='Training Iteration',
                                  ylabel='Loss')
                    )
            else:
                if not args.distributed or (
                    args.distributed and dist.get_rank() == 0):
                    vis.line(
                        X=np.arange(epoch*nbatches+train_iter, epoch*nbatches+train_iter+1),
                        Y=np.asarray([np.mean(train_loss)]),
                        win=vis_window['iter'],
                        opts=dict(title='Training Loss',
                                  xlabel='Training Iteration',
                                  ylabel='Loss'),
                        update='append'
                    )

        t_model_end = time.time()
        
        print('iter: [{}/{}], training loss: {:.4f}, '
              'grad norm: {:.4f} '
              'data time: {:.4f}s, total time: {:.4f}s'.format(
            train_iter, nbatches, cls_loss.item(),
            total_grad_norm,
            t_model_start - t_iter_start,
            t_model_end - t_iter_start
        ), end='\r')

        t_iter_start = time.time()

    return np.mean(train_loss)


def valid(model, loader):
    model.eval() # evaluation mode
    val_cls_loss = []
    for iter, data in enumerate(loader):
        (x_rpn_batch, obj_batch) = data

        x_rpn_batch = Variable(x_rpn_batch)
        obj_batch = Variable(obj_batch)

        if args.cuda:
            x_rpn_batch = x_rpn_batch.cuda()
            obj_batch = obj_batch.cuda()

        # N, C_out, T, num_class
        output, loss_weigh = model(x_rpn_batch, obj_batch)
        
        if args.loss_weighting or args.obj_interact: 
            rank_batch = F.margin_ranking_loss(output[:,0:1], output[:,1:2], Variable(
                torch.ones(output.size()).type(output.data.type())), margin=args.ranking_margin, reduce=False)
            if args.loss_weighting and args.obj_interact:
                loss_weigh = (output[:, 0:1]+loss_weigh)/2. # avg
            elif args.loss_weighting:
                loss_weigh = output[:,0:1]
            else:
                loss_weigh = loss_weigh.unsqueeze(1)
            # ranking loss
            cls_loss = args.loss_factor*(rank_batch*loss_weigh).mean()+ \
                        (1-args.loss_factor)*-torch.log(2*loss_weigh).mean()
        else:
            # ranking loss
            cls_loss = F.margin_ranking_loss(output[:,0:1], output[:,1:2], Variable(
                torch.Tensor([[1],[1]]).type(output.data.type())), margin=args.ranking_margin)

        val_cls_loss.append(cls_loss.item())
        if iter%100 == 0:
            print('evaluating sample # {}, val loss: {}'.format(iter, cls_loss.item()))

    return np.mean(val_cls_loss)


if __name__ == "__main__":
    main(args)
