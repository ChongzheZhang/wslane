#!/bin/bash
source /home/us6w74o/miniconda3/etc/profile.d/conda.sh;
conda activate wslane;
export CUDA_VISIBLE_DEVICES=1;
# Train
python ./main.py train ./configs/clrnet/clr_resnet18_cu2curve.py --work_dirs=clr_cu2curve_ema_lr5e5 --load_from=./outputs/clr_res18_culane_newseg_oriweight_b120/ckpt/15.pth --lr=5.0e-5