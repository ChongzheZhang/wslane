#!/bin/bash
source /home/us6w74o/miniconda3/etc/profile.d/conda.sh;
conda activate wslane;
export CUDA_VISIBLE_DEVICES=2;
# Train
python ./main.py train ./configs/laneatt/resnet18_tusimple.py --work_dirs=base1 --load_from=./outputs/laneatt_res18_culane_b40_seg1_allanchor_e25/ckpt/23.pth
# Test
# python ./main.py test ./configs/laneatt/resnet18_culane.py --work_dirs=test-e24 --load_from=./outputs/laneatt_res18_culane_b40_allanchor_e25/ckpt/24.pth
# python ./main.py test ./configs/laneatt/resnet18_tusimple.py --work_dirs=lanatt_res18_tusimple_b40 --load_from=./outputs/lanatt_res18_tusimple_b40/ckpt/86.pth
# python ./main.py test ./configs/laneatt/resnet18_tusimple.py --work_dirs=lanatt_res18_tusimple_b40 --load_from=./outputs/lanatt_res18_tusimple_b40/ckpt/92.pth
# python ./main.py test ./configs/laneatt/resnet18_tusimple.py --work_dirs=lanatt_res18_tusimple_b40 --load_from=./outputs/lanatt_res18_tusimple_b40/ckpt/96.pth
# python ./main.py test ./configs/laneatt/resnet18_tusimple.py --work_dirs=lanatt_res18_tusimple_b40 --load_from=./outputs/lanatt_res18_tusimple_b40/ckpt/100.pth