#!/bin/bash
source /home/us6w74o/miniconda3/etc/profile.d/conda.sh;
conda activate wslane;
export CUDA_VISIBLE_DEVICES=0;
# Train CLRNet
# python ./main.py train ./configs/clrnet/clr_erfnet_tusimple.py --work_dirs=clr_erfnet_tusimple_newseg_OsegW_b40
# python ./main.py train ./configs/clrnet/clr_erfnet_culane.py --work_dirs=clr_erfnet_culane_newseg_OsegW_b40
# python ./main.py train ./configs/clrnet/clr_resnet18_cu2tu.py --work_dirs=e3_num01 --load_from=./outputs/clr_res18_culane_newseg_oriweight_b120/ckpt/15.pth --num_lane_loss_weight=0.1 --no_comet
# python ./main.py train ./configs/clrnet/clr_resnet18_tu2cu.py --work_dirs=clr_tu2cu_final+dist --load_from=./outputs/clr_res18_tusimple_newseg_oriweight_b120/ckpt/70.pth
python ./main.py train ./configs/clrnet/clr_resnet18_cu2curve.py --work_dirs=clr_cu2curve_ema_lr1e4 --load_from=./outputs/clr_res18_culane_newseg_oriweight_b120/ckpt/15.pth --lr=1.0e-4

# Train LaneATT
# python ./main.py train ./configs/laneatt/resnet18_cu2tu.py --work_dirs=att_seg01 --load_from=./outputs/laneatt_res18_culane_b40_allanchor_newseg_nms15_segweight2/ckpt/25.pth --seg_loss_weight=0.1 --no_comet
# python ./main.py train ./configs/laneatt/resnet18_tusimple.py --work_dirs=laneatt_res18_tusimple_b40_allanchor_nms15_newseg_segW2_E80
# python ./main.py train ./configs/laneatt/resnet18_culane.py --work_dirs=laneatt_res18_culane_b40_allanchor_base
# python ./main.py train ./configs/laneatt/resnet18_cu2tu.py --work_dirs=debug --load_from=./outputs/laneatt_res18_culane_b40_allanchor_newseg_nms15_segweight2/ckpt/25.pth --no_comet
# python ./main.py train ./configs/laneatt/resnet18_tu2cu.py --work_dirs=laneatt_tu2cu_segweight2_E61_ema_reg06 --load_from=./outputs/laneatt_res18_tusimple_b40_allanchor_nms15_newseg_segW2_E80/ckpt/61.pth
# python ./main.py train ./configs/laneatt/resnet18_cu2curve.py --work_dirs=cu2curve_laneatt_base24 --load_from=./outputs/laneatt_res18_culane_b40_allanchor_seg1_nms15_segweight2/ckpt/24.pth

# Test
# python ./main.py test ./configs/laneatt/resnet18_tu2cu.py --work_dirs=debug1 --load_from=./outputs/laneatt_res18_tusimple_b40_allanchor_nms15_seg1_segweight1_E70/ckpt/43.pth --no_comet