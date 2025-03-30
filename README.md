To run this code, the configuration of a conda environment with all the required libraries is strongly recommended.

Required libraries and modules:
cudf, cupy, numpy, os, pandas, torch, imblearn, cuml, sklearn, typing, matplotlib, xgboost, optuna, joblib, pynvml, datetime, time, sys, argparse, seaborn, pytest


Please chech the following pages to install NVIDIA drivers and RAPIDS libraries in your environment.  
https://docs.rapids.ai/install/  
https://developer.nvidia.com/rapids/get-started  
https://www.nvidia.com/en-us/drivers/  
  
How to use this project:  
  
1. Download the dataset from Kaggle.com website (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and save it into the ccfd/data folder (name it creditcard.csv)  
2. All command line commands must be run from inside the ccfd folder (the one containing: ccfd, results, runs, scripts and tests folder).  
3. Pre-train the oversampling methods you want to use (only GAN and WGAN oversampling methods must be pre-trained).  
  e.g. python -m scripts.optimize_ovs --device gpu --ovs wgan --trials 30 --jobs 3  
4. Pre-train the ML models you want to use  
  e.g. python -m scripts.train_models --device gpu --model xgboost --ovs adasyn --output_folder ccfd/pretrained_models --metric recall --trials 30 --jobs 2  
  Training results are saved in the results folder, in csv format.  
5. Display training results  
   e.g. python -m scripts.display_results --file results/training_time_performance_2025_03_29.csv --sort recall  
   CSV files can also be loaded by any spreadsheet  
7. Test the model you pre-trained  
  e.g. python -m scripts.test_models --device gpu --model xgboost --ovs smote --metric prauc --eval_method cost_based --cost_fp 1 --cost_fn 3

