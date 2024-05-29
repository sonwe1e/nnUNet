export nnUNet_raw="/data2/songwei/Data/raw" 
export nnUNet_preprocessed="/data2/songwei/Data/processed/" 
export nnUNet_results="/data2/songwei/Data/test/" 
export nnUNet_outputs="/home/yhwu/sw/nnUNet/Saved_nnUNetv2/output/"
export PATH="/home/songwei/miniconda3/bin" 

# CUDA_VISIBLE_DEVICES=7 nnUNetv2_predict -i /data1/songwei/WorkStation/Data/raw/fold1 \
#  -o /data1/songwei/WorkStation/Data/results/Dataset011_AortaSeg/baseline+1kernelout/fold_1/validation_best \
#  -d 011 -c 3d_fullres -f 1 -ckp /data1/songwei/WorkStation/Data/results/Dataset011_AortaSeg/baseline+1kernelout/fold_1/checkpoint_best.pth \
#  -m /data1/songwei/WorkStation/Data/results/Dataset011_AortaSeg/baseline+1kernelout

# CUDA_VISIBLE_DEVICES=7 nnUNetv2_predict -i /data1/songwei/WorkStation/Data/raw/fold1 \
#  -o /data1/songwei/WorkStation/Data/results/Dataset011_AortaSeg/baseline-avgpool/fold_1/validation_best \
#  -d 011 -c 3d_fullres -f 1 -ckp /data1/songwei/WorkStation/Data/results/Dataset011_AortaSeg/baseline-avgpool/fold_1/checkpoint_final.pth \
#  -m /data1/songwei/WorkStation/Data/results/Dataset011_AortaSeg/baseline-avgpool

CUDA_VISIBLE_DEVICES=7 nnUNetv2_predict -i /data1/songwei/WorkStation/Data/raw/fold1 \
 -o /data1/songwei/WorkStation/Data/results/Dataset011_AortaSeg/baseline-truncnormalinit-avgpool/fold_1/validation_best \
 -d 011 -c 3d_fullres -f 1 -ckp /data1/songwei/WorkStation/Data/results/Dataset011_AortaSeg/baseline-truncnormalinit-avgpool/fold_1/checkpoint_final.pth \
 -m /data1/songwei/WorkStation/Data/results/Dataset011_AortaSeg/baseline-truncnormalinit-avgpool