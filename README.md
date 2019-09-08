# AIM2019-Alpha-ASSR
AIM 2019 Constrained Super-Resolution Challenge, Alpha Team, ASSR

This code is built on [EDSR](https://github.com/thstkdgus35/EDSR-PyTorch) (PyTorch)

1. Put the DIV2K test LR image into test folder.(already exists)
2. run 
```bash
cd src       # You are now in */AIM2019-Alpha-ASSR/src
python main.py --test_only --data_test Demo --scale 4 --model assr --pre_train ../models/ASSR_x4_best.pt --save_results
```
3. You can find the result images from ```experiment/test/results-Demo``` folder.
