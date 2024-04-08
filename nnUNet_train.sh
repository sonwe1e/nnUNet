export nnUNet_raw="/home/yhwu/sw/nnUNet/Saved_nnUNetv2/raw" 
export nnUNet_preprocessed="/home/yhwu/sw/nnUNet/Saved_nnUNetv2/processed/" 
export nnUNet_results="/home/yhwu/sw/nnUNet/Saved_nnUNetv2/test/" 
export PATH="/home/yhwu/miniconda3/envs/sw/bin" 


# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 501 3d_fullres_smallpatch_finetune 2 --npz -tr nnUNetTrainer_500epochs_5e_4lr \
#  -pretrained_weights /home/yhwu/sw/Saved_nnUNetv2/results/Dataset003_IXI/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_best.pth &
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 501 3d_fullres 2 --npz -tr MyTrainer --exp_name mynet_skdoubleconv_mschead
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 007 3d_fullres 2 --npz -tr DodTrainer --exp_name test