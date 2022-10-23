
exp_name=motion_iris
rm -rf logs/$exp_name
rm -rf checkpoints/$exp_name
CUDA_VISIBLE_DEVICES=5 taskset --cpu-list 20-30 python3 train_iris.py --viz_output viz_$exp_name\
                                      --exp_name $exp_name\
                                      --num_kp 2
