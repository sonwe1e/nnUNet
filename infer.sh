export nnUNet_raw="/data2/songwei/Data/raw" 
export nnUNet_preprocessed="/data2/songwei/Data/processed/" 
export nnUNet_results="/data2/songwei/Data/test/" 
export nnUNet_outputs="/home/yhwu/sw/nnUNet/Saved_nnUNetv2/output/"
export PATH="/home/yhwu/miniconda3/envs/umamba/bin" 

CUDA_VISIBLE_DEVICES=3 nnUNetv2_predict -i /data2/songwei/Data/LargeIA_left/imagesTs \
 -o /data2/songwei/Data/test/Dataset010_IXI_LargeIA_ALL/dodnet_mschead_1000_0.66_192/fold_all/validation_best \
 -d 010 -c 3d_fullres_big_patch -f all -tr DodTrainer -ckp /data2/songwei/Data/test/Dataset010_IXI_LargeIA_ALL/dodnet_mschead_1000_0.66_192/fold_all/checkpoint_best.pth \
 -m /data2/songwei/Data/test/Dataset010_IXI_LargeIA_ALL/dodnet_baseline_1000_0.66_192 &

CUDA_VISIBLE_DEVICES=3 nnUNetv2_predict -i /data2/songwei/Data/LargeIA_left/imagesTsCroped \
 -o /data2/songwei/Data/test/Dataset010_IXI_LargeIA_ALL/dodnet_mschead_1000_0.66_192/fold_all/validation_crop \
 -d 010 -c 3d_fullres_big_patch -f all -tr DodTrainer -ckp /data2/songwei/Data/test/Dataset010_IXI_LargeIA_ALL/dodnet_mschead_1000_0.66_192/fold_all/checkpoint_best.pth \
 -m /data2/songwei/Data/test/Dataset010_IXI_LargeIA_ALL/dodnet_baseline_1000_0.66_192 &

# CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict -i /data2/songwei/Data/LargeIA_left/imagesTsCroped \
#  -o /data2/songwei/Data/test/Dataset010_IXI_LargeIA_ALL/dodnet_baseline_2000_0.66_192/fold_all/validation_crop \
#  -d 010 -c 3d_fullres_big_patch -f all -tr DodTrainer -ckp /data2/songwei/Data/test/Dataset010_IXI_LargeIA_ALL/dodnet_baseline_2000_0.66_192/fold_all/checkpoint_best.pth \
#  -m /data2/songwei/Data/test/Dataset010_IXI_LargeIA_ALL/dodnet_baseline_1000_0.66_192 &

# CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict -i /data2/songwei/Data/LargeIA_left/imagesTsCroped \
#  -o /data2/songwei/Data/test/Dataset006_LargeIA_Croped/mynet_baseline_1000_0.66_192/fold_all/validation_crop \
#  -d 006 -c 3d_fullres_big_patch -f all -tr MyTrainer -ckp /data2/songwei/Data/test/Dataset006_LargeIA_Croped/mynet_baseline_1000_0.66_192/fold_all/checkpoint_best.pth \
#  -m /data2/songwei/Data/test/Dataset006_LargeIA_Croped/nnunet_1000_0.66_128 &

# CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i /data2/songwei/Data/LargeIA_left/imagesTsCroped \
#  -o /data2/songwei/Data/test/Dataset006_LargeIA_Croped/nnunet_1000_0.66_128/fold_all/validation_crop \
#  -d 006 -c 3d_fullres -f all  -ckp /data2/songwei/Data/test/Dataset006_LargeIA_Croped/nnunet_1000_0.66_128/fold_all/checkpoint_best.pth \
#  -m /data2/songwei/Data/test/Dataset006_LargeIA_Croped/nnunet_1000_0.66_128