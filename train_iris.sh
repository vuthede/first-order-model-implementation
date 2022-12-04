
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



# exp_name=motion_iris_fix_motion_test_zero_order_motion
# rm -rf logs/$exp_name
# rm -rf checkpoints/$exp_name
# rm -rf viz_$exp_name
# CUDA_VISIBLE_DEVICES=5 taskset --cpu-list 20-30 python3 train_iris.py --viz_output viz_$exp_name\
#                                       --exp_name $exp_name\
#                                       --num_kp 2\
#                                       --using_first_order_motion 0


# exp_name=motion_iris_thin_plate_spline_motion
# rm -rf logs/$exp_name
# rm -rf checkpoints/$exp_name
# rm -rf viz_$exp_name
# CUDA_VISIBLE_DEVICES=5 taskset --cpu-list 20-30 python3 train_iris.py --viz_output viz_$exp_name\
#                                       --exp_name $exp_name\
#                                       --num_kp 2\
#                                       --using_first_order_motion 0\
#                                       --using_thin_plate_spline_motion 1

exp_name=motion_iris_thin_plate_spline_motion_more_control_points
rm -rf logs/$exp_name
rm -rf checkpoints/$exp_name
rm -rf viz_$exp_name
CUDA_VISIBLE_DEVICES=5 taskset --cpu-list 20-30 python3 train_iris.py --viz_output viz_$exp_name\
                                      --exp_name $exp_name\
                                      --num_kp 8\
                                      --using_first_order_motion 0\
                                      --using_thin_plate_spline_motion 1


# exp_name=motion_iris_thin_plate_with_eyelid_motion
# rm -rf logs/$exp_name
# rm -rf checkpoints/$exp_name
# rm -rf viz_$exp_name
# CUDA_VISIBLE_DEVICES=5 taskset --cpu-list 20-30 python3 train_iris.py --viz_output viz_$exp_name\
#                                       --exp_name $exp_name\
#                                       --num_kp 8\
#                                       --using_first_order_motion 0\
#                                       --using_thin_plate_spline_motion 1\
#                                       --estimate_lid_motion 1
