#!/bin/bash


##################
################### 01 classical sr

### train
############### SWIN
# torchrun --nproc_per_node=2 --master_port=1234 main_train_psnr_re.py \
#     --opt options/swinir/train_swinir_sr_classical.json --dist True


############### EQ-reg x2
# torchrun --nproc_per_node=2 --master_port=12346 main_train_psnr_loss.py \
#     --opt options/swinir/train_swinir_sr_classical_loss.json --dist True

############## EQ-reg x3
# torchrun  --nproc_per_node=4 --master_port=1234 main_train_psnr_loss.py \
#     --opt options/swinir/train_swinir_sr_classical_loss_x3.json --dist True

############## EQ-reg x4
# torchrun  --nproc_per_node=4 --master_port=12347 main_train_psnr_loss.py \
#     --opt options/swinir/train_swinir_sr_classical_loss_x4.json --dist True



### Test
############### SWIN
# python main_test_swinir.py --task classical_sr --scale 2 --training_patch_size 48 \
#     --model_path path/to/pretrained_models.pth --folder_lq path/to/your/datasets --folder_gt path/to/your/datasets

############### EQ-reg
# python main_test_swinir_loss.py --task classical_sr --scale 2 --training_patch_size 48 \
#     --model_path path/to/pretrained_models.pth --folder_lq path/to/your/datasets --folder_gt path/to/your/datasets




##################
################### color denoising

### train
################ SWIN
# torchrun  --nproc_per_node=4 --master_port=1234 main_train_psnr_re.py \
#         --opt options/swinir/train_swinir_denoising_color.json  --dist True

############### EQ-reg
# torchrun  --nproc_per_node=8 --master_port=12346 main_train_psnr_loss.py \
#         --opt options/swinir/train_swinir_denoising_color_loss.json  --dist True





### Test
############## SWIN
# python main_test_swinir.py --task color_dn --noise 50 \
#             --model_path path/to/pretrained_models.pth --folder_gt path/to/your/datasets

############### EQ-reg
# python main_test_swinir_loss.py --task color_dn --noise 50 \
#             --model_path path/to/pretrained_models.pth --folder_gt path/to/your/datasets
       
