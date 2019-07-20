# FCOS_PLUS

This project contains some improvements about FCOS (Fully Convolutional One-Stage Object Detection).


## Installation

Please check [INSTALL.md](INSTALL.md) (same as original FCOS) for installation instructions. 


**Results**


Model | Total training mem (GB) | Multi-scale training | Testing time / im | AP (minival) 
---   |:---:|:---:|:---:|:---:|
FCOS_R_50_FPN_1x | 29.3 | No | 71ms | 37.0 |
FCOS_R_50_FPN_1x_center | 30.61 | No | 71ms | 37.8 |
FCOS_R_50_FPN_1x_center_liou | 30.61 | No | 71ms | 38.1 |
FCOS_R_50_FPN_1x_center_giou | 30.61 | No | 71ms | 38.2 | 
FCOS_R_101_FPN_2x | 44.1 | Yes | 74ms | 41.4 |
FCOS_R_101_FPN_2x_center_giou | 44.1 | Yes | 74ms | 42.5 |

[1] *1x and 2x mean the model is trained for 90K and 180K iterations, respectively.* \
[2] center mean [center sample](fcos.pdf) are used in our training. \
[3] liou mean the model use linear iou loss function. (1 - iou) \
[4] giou mean the use giou loss function. (1 - giou) 


## Training

The following command line will train FCOS_R_50_FPN_1x on 8 GPUs with Synchronous Stochastic Gradient Descent (SGD):

    python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port=$((RANDOM + 10000)) \
        tools/train_net.py \
        --skip-test \
        --config-file configs/fcos/fcos_R_50_FPN_1x_center_giou.yaml \
        DATALOADER.NUM_WORKERS 2 \
        OUTPUT_DIR training_dir/fcos_R_50_FPN_1x_center_giou
        
Note that:
1) If you want to use fewer GPUs, please change `--nproc_per_node` to the number of GPUs. No other settings need to be changed. The total batch size does not depends on `nproc_per_node`. If you want to change the total batch size, please change `SOLVER.IMS_PER_BATCH` in [configs/fcos/fcos_R_50_FPN_1x_center_giou.yaml](configs/fcos/fcos_R_50_FPN_1x_center_giou.yaml).
2) The models will be saved into `OUTPUT_DIR`.
3) If you want to train FCOS with other backbones, please change `--config-file`.

## Citations
Please consider citing original paper in your publications if the project helps your research. 
```
@article{tian2019fcos,
  title   =  {{FCOS}: Fully Convolutional One-Stage Object Detection},
  author  =  {Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  journal =  {arXiv preprint arXiv:1904.01355},
  year    =  {2019}
}
```


## License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact the authors. 

