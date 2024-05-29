export nnUNet_raw="/data1/songwei/WorkStation/Data/raw" 
export nnUNet_preprocessed="/data1/songwei/WorkStation/Data/processed/" 
export nnUNet_results="/data1/songwei/WorkStation/Data/results/" 
export PATH="/home/songwei/miniconda3/bin" 

# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 011 3d_fullres_bigpatch 1 -tr MyTrainer --npz --exp_name baseline-224x224x224
# CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 011 3d_fullres 1 -tr MyTrainer --npz --exp_name baseline-1x3x5conv
# CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 011 3d_fullres 1 -tr MyTrainer --npz --exp_name baseline-avgpool
# CUDA_VISIBLE_DEVICES=4 nnUNetv2_train 011 3d_fullres 1 -tr MyTrainer --npz --exp_name baseline+1updoubleconv
# CUDA_VISIBLE_DEVICES=5 nnUNetv2_train 011 3d_fullres 1 -tr MyTrainer --npz --exp_name baseline-truncnormalinit

CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 011 3d_fullres 1 -tr MyTrainer --npz --exp_name baseline-truncnormalinit-avgpool