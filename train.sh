export nnUNet_raw="/home/yhwu/sw/Dataset/raw" 
export nnUNet_preprocessed="/home/yhwu/sw/Dataset/processed/" 
export nnUNet_results="/home/yhwu/sw/nnUNet/Saved_nnUNetv2/test/" 
export PATH="/home/yhwu/miniconda3/envs/sw/bin" 


# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 501 3d_fullres_smallpatch_finetune 2 --npz -tr nnUNetTrainer_500epochs_5e_4lr \
#  -pretrained_weights /home/yhwu/sw/Saved_nnUNetv2/results/Dataset003_IXI/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_best.pth &
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 501 3d_fullres 2 --npz -tr MyTrainer --exp_name mynet_skdoubleconv_mschead
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 007 3d_fullres 1 --npz -tr DodTrainer --exp_name test_30IXI_mynet_dod_adam2fold
# CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 501 3d_fullres_dod 2 --npz -tr DodTrainer --exp_name ADAM_nnUNet_fold2