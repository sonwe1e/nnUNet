export nnUNet_raw="/data2/songwei/Data/raw" 
export nnUNet_preprocessed="/data2/songwei/Data/processed/" 
export nnUNet_results="/data2/songwei/Data/test/" 
# export PATH="/home/yhwu/miniconda3/envs/sw/bin" 
export PATH="/home/yhwu/miniconda3/envs/umamba/bin" 

# CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 009 3d_fullres_small_patch 0 --npz -tr DodTrainer --exp_name dodnet_loss_weight_10_1_0.66_1000_64
# CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 009 3d_fullres_big_patch 0 --npz -tr DodTrainer --exp_name dodnet_loss_weight_10_1_0.66_500_192 
# CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 009 3d_fullres 0 --npz -tr DodTrainer --exp_name dodnet_loss_weight_10_1_0.66_mschead_fp32

# CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 006 3d_fullres_big_patch all --npz --exp_name nnunet_1000_0.66_192 & 
# CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 006 3d_fullres_big_patch all -tr MyTrainer --npz --exp_name mynet_baseline_1000_0.66_192

# CUDA_VISIBLE_DEVICES=4 nnUNetv2_train 010 3d_fullres_big_patch all -tr DodTrainer --npz --exp_name dodnet_baseline_1000_0.66_192_0.95_0.05
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 010 3d_fullres_big_patch all -tr DodTrainer --npz --exp_name dodnet_embedding_1000_0.66_192_0.95_0.05
CUDA_VISIBLE_DEVICES=4 nnUNetv2_train 010 3d_fullres_big_patch all -tr DodTrainer --npz --exp_name dodnet_embedding

# CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 011 3d_fullres 1 --npz --exp_name nnunet_1000_0.5_deep_volume_min
# CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 011 3d_fullres 2 --npz --exp_name nnunet_1000_0.33_deep &
# CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 011 3d_fullres 3 --npz --exp_name nnunet_1000_0.33_deep &
# CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 011 3d_fullres 4 --npz --exp_name nnunet_1000_0.33_deep &
# CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 011 3d_fullres 0 --npz --exp_name nnunet_1000_0.33_deep