
######### Default full 
# rm -rf logs/baseline_vox_first_order_motion
# sleep 3
# CUDA_VISIBLE_DEVICES=4 taskset --cpu-list 0-20 python3 train.py

######## Remove the kp detection
# exp_name=wo_equivariance_kp
# rm -rf logs/$exp_name
# rm -rf checkpoints/$exp_name
# CUDA_VISIBLE_DEVICES=5 taskset --cpu-list 20-30 python3 train.py --viz_output viz_$exp_name\
#                                             --lamda_equi_value_loss 0.0\
#                                             --lamda_equi_jacobian_loss 0.0


######## Remove the vgg19
# exp_name=wo_vgg19_rec
# rm -rf logs/$exp_name
# rm -rf checkpoints/$exp_name
# CUDA_VISIBLE_DEVICES=6 taskset --cpu-list 30-40 python3 train.py --viz_output viz_$exp_name\
#                                             --lamda_rec_vgg 0.0\


######## Using zeros order motion instead of first order motion. Just let jacobian ==[1 0 0 1]
# exp_name=zero_order_motion
# rm -rf logs/$exp_name
# rm -rf checkpoints/$exp_name
# CUDA_VISIBLE_DEVICES=7 taskset --cpu-list 40-50 python3 train.py --viz_output viz_$exp_name\
#                                             --using_first_order_motion 0\

exp_name=wo_equivariance_jacobian
rm -rf logs/$exp_name
rm -rf checkpoints/$exp_name
CUDA_VISIBLE_DEVICES=4 taskset --cpu-list 0-20 python3 train.py --viz_output viz_$exp_name\
                                            --lamda_equi_jacobian_loss 0.0\
