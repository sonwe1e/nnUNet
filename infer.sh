export nnUNet_raw="/home/yhwu/sw/Dataset/raw" 
export nnUNet_preprocessed="/home/yhwu/sw/Dataset/processed/" 
export nnUNet_results="/home/yhwu/sw/nnUNet/Saved_nnUNetv2/test/" 
export nnUNet_outputs="/home/yhwu/sw/nnUNet/Saved_nnUNetv2/output/"
export PATH="/home/yhwu/miniconda3/envs/sw/bin" 

# CUDA_VISIBLE_DEVICES=3 nnUNetv2_predict -i /home/yhwu/sw/nnUNet/Saved_nnUNetv2/test/Dataset007_IXI_ADAM/test/fold_1/validation_raw\
#  -o /home/yhwu/sw/nnUNet/Saved_nnUNetv2/test/Dataset007_IXI_ADAM/test/fold_1/validation_taskIXI\
#  -d 501 -c 3d_fullres -f 2 -tr DodTrainer -ckp /home/yhwu/sw/nnUNet/Saved_nnUNetv2/test/Dataset007_IXI_ADAM/test_30IXI_mynet_dod/fold_1/checkpoint_best.pth\
#  -m /home/yhwu/sw/nnUNet/Saved_nnUNetv2/test/Dataset007_IXI_ADAM/test

# CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict -i /home/yhwu/sw/nnUNet/Saved_nnUNetv2/test/Dataset007_IXI_ADAM/test/fold_1/validation_raw\
#  -o /home/yhwu/sw/nnUNet/Saved_nnUNetv2/test/Dataset007_IXI_ADAM/test/fold_1/validation_taskADAM\
#  -d 501 -c 3d_fullres -f 2 -tr DodTrainer -ckp /home/yhwu/sw/nnUNet/Saved_nnUNetv2/test/Dataset007_IXI_ADAM/test_noIXI/fold_1/checkpoint_best.pth\
#  -m /home/yhwu/sw/nnUNet/Saved_nnUNetv2/test/Dataset007_IXI_ADAM/test

CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i /home/yhwu/sw/nnUNet/Saved_nnUNetv2/results/Dataset501_ADAM/mynet_resnext_mschead_500_2e-4/fold_2/validation_raw\
 -o /home/yhwu/sw/nnUNet/Saved_nnUNetv2/test/Dataset007_IXI_ADAM/test_30IXI_mynet_dod_adam2fold/fold_1/validation_ADAM_fold2\
 -d 501 -c 3d_fullres -f 2  -ckp /home/yhwu/sw/nnUNet/Saved_nnUNetv2/test/Dataset007_IXI_ADAM/test_30IXI_mynet_dod_adam2fold/fold_1/checkpoint_best.pth\
 -m /home/yhwu/sw/nnUNet/Saved_nnUNetv2/test/Dataset007_IXI_ADAM/test_30IXI_mynet_dod_adam2fold