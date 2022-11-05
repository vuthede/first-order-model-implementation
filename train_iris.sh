
# exp_name=motion_iris
# rm -rf logs/$exp_name
# rm -rf checkpoints/$exp_name
# CUDA_VISIBLE_DEVICES=5 taskset --cpu-list 20-30 python3 train_iris.py --viz_output viz_$exp_name\
#                                       --exp_name $exp_name\
#                                       --num_kp 2



# exp_name=motion_iris_fix_motion_equation
# rm -rf logs/$exp_name
# rm -rf checkpoints/$exp_name
# rm -rf viz_$exp_name
# CUDA_VISIBLE_DEVICES=5 taskset --cpu-list 20-30 python3 train_iris.py --viz_output viz_$exp_name\
#                                       --exp_name $exp_name\
#                                       --num_kp 2



exp_name=motion_iris_fix_motion_test_zero_order_motion
rm -rf logs/$exp_name
rm -rf checkpoints/$exp_name
rm -rf viz_$exp_name
CUDA_VISIBLE_DEVICES=5 taskset --cpu-list 20-30 python3 train_iris.py --viz_output viz_$exp_name\
                                      --exp_name $exp_name\
                                      --num_kp 2\
                                      --using_first_order_motion 0
