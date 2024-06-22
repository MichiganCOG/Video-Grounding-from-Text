# Weakly-Supervised Video Object Grounding from Text by Loss Weighting and Object Interaction

This is the source code from our paper [Weakly-Supervised Video Object Grounding from Text by Loss Weighting and Object Interaction](http://bmvc2018.org/contents/papers/0070.pdf)


## Requirements (Recommended)
1) CUDA 9.0 and CUDNN v7.1

2) Install [Miniconda](https://conda.io/miniconda.html) (either Miniconda2 or 3, version 4.6+). We recommend using conda [environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to install required packages, including Python 3.6, [PyTorch 0.4.0](https://pytorch.org/get-started/locally/) etc.:
```
MINICONDA_ROOT=[to your Miniconda root directory]
conda env create -f tools/conda_env_yc2_bb.yml --prefix $MINICONDA_ROOT/envs/yc2-bb
conda activate yc2-bb
python -m spacy download en # to download spacy English model
```


## Data Preparation
Download the followings.

1) The YouCook2-BB annotation pack from the official [website](http://youcook2.eecs.umich.edu/download),
**[06/22/2024]** Due to requests and inaccessibility of online videos, we are now sharing the raw video files for **non-commercial, research purposes only**. They can be found in Download pages of the offical website.

3) Region proposals [[all-in-one](http://youcook2.eecs.umich.edu/static/dat/yc2_bb/all-box-100.tar.gz)] and feature files for each split [[train(113GB)](http://youcook2.eecs.umich.edu/static/dat/yc2_bb/roi_pooled_feat_train.tar.gz), [val(38GB)](http://youcook2.eecs.umich.edu/static/dat/yc2_bb/roi_pooled_feat_val.tar.gz), [test(17GB)](http://youcook2.eecs.umich.edu/static/dat/yc2_bb/roi_pooled_feat_test.tar.gz)]. You can also extract features/proposals on your own using Faster RCNN [PyTorch](https://github.com/LuoweiZhou/faster-rcnn.pytorch).

Place all the downloaded files under `data/yc2` and uncompress.


## Running
### Training
The example command on running a 4-GPU distributed data parallel job:
```
CUDA_VISIBLE_DEVICES=0 python train.py --loss_weighting --obj_interact --checkpoint_path $checkpoint_path --cuda --world_size 4 &
CUDA_VISIBLE_DEVICES=1 python train.py --loss_weighting --obj_interact --checkpoint_path $checkpoint_path --cuda --world_size 4 &
CUDA_VISIBLE_DEVICES=2 python train.py --loss_weighting --obj_interact --checkpoint_path $checkpoint_path --cuda --world_size 4 &
CUDA_VISIBLE_DEVICES=3 python train.py --loss_weighting --obj_interact --checkpoint_path $checkpoint_path --cuda --world_size 4
``` 
(Optional) Set `--world_size 1` to run in single-GPU mode.

(Optional) To visualize the training curves, we use `visdom` (install through `pip install visdom`). Start the server (probably in a `tmux` or `screen`) in the background with the command: `visdom`. In your training command, add `--enable_visdom` as a command argument.

### Testing
You can download the pre-trained model from [here](http://youcook2.eecs.umich.edu/static/dat/yc2_bb/full-model.pth) (`model_checkpoint=full-model.pth`) and place it under the `checkpoint` dir.
```
python test.py --start_from ./checkpoint/$model_checkpoint --val_split validation --cuda
``` 
The evaluation server on the test set is now available on [Codalab](https://competitions.codalab.org/competitions/20302)!


### Visualization
This requires opencv2 and can be done by running command `conda install -c menpo opencv`.
```
python vis.py --start_from ./checkpoint/$model_checkpoint --cuda
```


## Notes
After releasing the original version of YouCook2-BB, we have added 149/4316=3.5% more annotations to the dataset. As a result, the overall model performance has had a slight change: 30.1% now v.s. 30.3% before on the validation set and 32.0% now v.s. 31.7% before on the test set.


## Citation
Please acknowledge the following paper if you use the code:
```
  @inproceedings{ZhLoCoBMVC18,
    author={Zhou, Luowei and Louis, Nathan and Corso, Jason J},
    title={Weakly-Supervised Video Object Grounding from Text by Loss Weighting and Object Interaction},
    booktitle = {British Machine Vision Conference},
    year = {2018},
    url = {http://bmvc2018.org/contents/papers/0070.pdf}
  }
```
