export nnUNet_raw="/home/yhwu/sw/nnUNet/Saved_nnUNetv2/raw" 
export nnUNet_preprocessed="/home/yhwu/sw/nnUNet/Saved_nnUNetv2/processed/" 
export nnUNet_results="/home/yhwu/sw/nnUNet/Saved_nnUNetv2/test/" 
export PATH="/home/yhwu/miniconda3/envs/sw/bin" 

# nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres -np 64 --verify_dataset_integrity --verbose
# nnUNetv2_plan_experiment -d 003

nnUNetv2_plan_and_preprocess -d 007 -c 3d_fullres --verify_dataset_integrity --verbose