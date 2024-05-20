export nnUNet_raw="/data2/songwei/Data/raw" 
export nnUNet_preprocessed="/data2/songwei/Data/processed/" 
export nnUNet_results="/data2/songwei/Data/test/" 
export PATH="/home/yhwu/miniconda3/envs/umamba/bin" 

# nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres -np 64 --verify_dataset_integrity --verbose
# nnUNetv2_plan_experiment -d 003

nnUNetv2_plan_and_preprocess -d 007 -c 3d_fullres --verify_dataset_integrity --verbose
# nnUNetv2_plan_and_preprocess -d 011 -c 3d_fullres --verify_dataset_integrity --verbose