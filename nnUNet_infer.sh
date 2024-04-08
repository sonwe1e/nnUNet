export nnUNet_raw="/home/yhwu/sw/nnUNet/Saved_nnUNetv2/raw" 
export nnUNet_preprocessed="/home/yhwu/sw/nnUNet/Saved_nnUNetv2/processed/" 
export nnUNet_results="/home/yhwu/sw/nnUNet/Saved_nnUNetv2/test/" 
export nnUNet_outputs="/home/yhwu/sw/nnUNet/Saved_nnUNetv2/output/"
export PATH="/home/yhwu/miniconda3/envs/sw/bin" 

CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict -i /home/yhwu/sw/nnUNet/Saved_nnUNetv2/results/Dataset501_ADAM/mynet_resnext_mschead_500_2e-4/fold_2/validation_raw\
 -o /home/yhwu/sw/nnUNet/Saved_nnUNetv2/test/Dataset007_IXI_ADAM/test/fold_2/validation\
 -d 501 -c 3d_fullres -f 2 -tr DodTrainer -ckp /home/yhwu/sw/nnUNet/Saved_nnUNetv2/test/Dataset007_IXI_ADAM/test/fold_2/checkpoint_best.pth\
 -m /home/yhwu/sw/nnUNet/Saved_nnUNetv2/test/Dataset007_IXI_ADAM/test