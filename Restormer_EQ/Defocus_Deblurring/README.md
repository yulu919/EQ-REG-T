## Training

- To download DPDD training data, run
```
python download_data.py --data train
```

- Generate image patches from full-resolution training images, run
```
python generate_patches_dpdd.py 
```

- To train Restormer and Restormer-reg on **single-image** defocus deblurring task, run
```
cd Restormer_EQ
./train.sh Defocus_Deblurring/Options/DefocusDeblur_Single_8bit_Restormer.yml

./train.sh Defocus_Deblurring/Options/DefocusDeblur_Single_8bit_Restormer_loss.yml
```

- To train Restormer and Restormer-reg on **dual-pixel** defocus deblurring task, run
```
cd Restormer_EQ
./train.sh Defocus_Deblurring/Options/DefocusDeblur_DualPixel_16bit_Restormer.yml

./train.sh Defocus_Deblurring/Options/DefocusDeblur_DualPixel_16bit_Restormer_loss.yml
```


## Evaluation


- Download test dataset, run
```
python download_data.py --data test
```



- Testing on **single-image** defocus deblurring task, run
```
python test_single_image_defocus_deblur.py --save_images
```

- Testing on **dual-pixel** defocus deblurring task, run
```
python test_dual_pixel_defocus_deblur.py --save_images
```

The above testing scripts will reproduce image quality scores of Table 3 in the paper. 
