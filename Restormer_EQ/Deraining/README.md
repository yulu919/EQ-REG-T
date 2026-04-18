
## Training

1. To download Rain13K training and testing data, run
```
python download_data.py --data train-test
```

2. To train Restormer and Restormer-reg with default settings, run
```
cd Restormer
./train.sh Deraining/Options/Deraining_Restormer.yml

./train.sh Deraining/Options/Deraining_Restormer_loss.yml
```

## Evaluation

Download test datasets (Test100, Rain100H, Rain100L, Test1200, Test2800), run 
```
python download_data.py --data test
```

3. Testing
```
python test.py
```

#### To reproduce PSNR/SSIM scores of Table 1, run

```
evaluate_PSNR_SSIM.m 
```
