# RoGSplat: Learning Robust Generalizable Human Gaussian Splatting from Sparse Multi-View Images
## [Paper](https://arxiv.org/abs/2503.14198)
This is official code of CVPR2025 paper [RoGSplat: Learning Robust Generalizable Human Gaussian Splatting from Sparse Multi-View Images](https://arxiv.org/abs/2503.14198).

## Installation

### Set up python environment

```commandline
sh ./install.sh
```

### Set up dataset
#### RenderPeople
Please follow [sherf](https://github.com/skhu101/SHERF) to download the RenderPeople dataset. Unzip and rename it into ```renderpeople```, split dataset into ```train``` and ```val```.

#### Preprocessed SMPL
Download our estimated and preprocessed data from [here](https://1drv.ms/u/c/977dca105dd7d644/EduU7vMguu5EskJzEFFnPrMBFu-bRoy0933JQE0ra3OmXQ?e=zjggTs) and put it under ```renderpeople``` folder.

The final "renderpeople" folder should be like this:
```
renderpeople
  ├-- train
  ├-- val
  ├-- easymocap_smpl
```

Make a soft link:
```commandline
ln -s /path/to/renderpeople ./renderpeople
```

## Trainning
Train first stage:
```commandline
python train_stage1.py
```
Then change [stage1_ckpt](config/config_rp.yaml#L2) to ```./experiments/rp_xxxx/ckpt/rp_final.pth```, change [depth_ckpt](config/config_rp.yaml#L4) to ```./experiments/rp_xxxx/ckpt/rp_depth_latest.pth```.

Then train second stage:
```commandline
python train_stage2.py
```

## Evaluation
```commandline
python test_stage2.py --ckpt_path experiments/rp_xxxx
```

## Citation
If you find this code useful for your research, please cite this:
```
@inproceedings{RoGSplat2025CVPR,
    title={{RoGSplat}: Learning Robust Generalizable Human Gaussian Splatting from Sparse Multi-View Images},
    author={Xiao, Junjin and Zhang, Qing and Nie, Yongwei and Zhu, Lei and Zheng, Wei-Shi},
    booktitle={CVPR},
    year={2025}
}
```