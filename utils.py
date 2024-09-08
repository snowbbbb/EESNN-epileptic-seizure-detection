
import os 
import numpy as np 
import argparse
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import pandas as pd 
from sklearn.metrics import confusion_matrix

ch_names_list = ['EEG Fp1-Ref','EEG Fp2-Ref','EEG F3-Ref','EEG F4-Ref','EEG C3-Ref', 'EEG C4-Ref',
 'EEG P3-Ref', 'EEG P4-Ref', 'EEG O1-Ref', 'EEG O2-Ref', 'EEG F7-Ref', 'EEG F8-Ref', 
 'EEG T3-Ref', 'EEG T4-Ref', 'EEG T5-Ref', 'EEG T6-Ref', 'EEG Fz-Ref', 'EEG Cz-Ref', 'EEG Pz-Ref', 
 'POL E', 'POL PG1', 'POL PG2', 'EEG A1-Ref', 'EEG A2-Ref', 'POL T1', 'POL T2',
  'POL X1', 'POL X2', 'POL X3', 'POL X4', 'POL X5', 'POL X6', 'POL X7', 'POL SpO2', 'POL EtCO2', 
'POL DC03', 'POL DC04', 'POL DC05', 'POL DC06', 'POL Pulse', 'POL CO2Wave', 'POL $A1', 'POL $A2']


def set_cnn_model_parameters():
    parser = argparse.ArgumentParser(description=' get cnn model parameters')

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--model_id',type=int,default=1)
    parser.add_argument('--dataset', type=str, default='pku') # is pku before 2022.1.14
    parser.add_argument('--model_name', type=str, default='cw_srnet')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--num_cols',type=int,default=18)
    #parser.add_augument('--cuda', type=str, default='0')

    parser.add_argument('--time_window_size',type=int, default=2,help='time window_size')# default=2,help='time window_size')
    parser.add_argument('--frequency',type=int, default=50, help='data frequency ')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--use_scale',type=int,default=0)
    parser.add_argument('--use_pretrain',type=float,default=0.2)

    parser.add_argument('--use_print',type=int,default=0,help='if use print in model')
    parser.add_argument('--lr',type=float, default=1e-8)
    parser.add_argument('--lam',type=float, default=1e-3)
    parser.add_argument('--info', type=str, default='')
    parser.add_argument('--time',type=str,default='')
    parser.add_argument('--seed', type=int,default=42)
    parser.add_argument('--print_code',type=int, default=0)
    parser.add_argument('--fold_n',type=int,default=1)
    parser.add_argument('--flag',type=str,default='')
  #  parser.add_argument('--flag',type=str,default='')
    parser.add_argument('--optimizer',type=str,default='Adam')
    parser.add_argument('--idenNum',type=int,default=18)
   # parser.add_argument('--DANNType',type=int,default=1)

    return parser.parse_args()


global_params = set_cnn_model_parameters()

def use_cuda(x,cuda_id='1'):
    #device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
    cuda_id = global_params.cuda

    torch.cuda.set_device(cuda_id)
    return x.cuda()
    print('use cpu, not gpu ')
    return x
'''
True Positive(TP)：预测为正，判断正确；
False Positive(FP)：预测为正，判断错误；
True Negative(TN)：预测为负，判断正确；
False Negative(FN)：预测为负，判断错误。
- precision = TP / (TP + FP)
- recall = TP / (TP + FN)
- accuracy = (TP + TN) / (TP + FP + TN + FN)
- error rate =  (FN + FP) / (TP + FP + TN + FN)
- F1 Score = 2*P*R/(P+R)，其中P和R分别为 precision 和 recall
'''


# accuracy_score,recall_score,f1_score
def evaluate_result(true_y, predict_y):
    N = len(true_y)
    TP = sum([1 if (true_y[i]==1 and predict_y[i]==1) else 0 for i in range(N)])
    FP = sum([1 if (true_y[i]==0 and predict_y[i]==1) else 0 for i in range(N)])
    TN = sum([1 if (true_y[i]==0 and predict_y[i]==0) else 0 for i in range(N)])
    FN = sum([1 if (true_y[i]==1 and predict_y[i]==0) else 0 for i in range(N)])
    precision = TP /(TP+FP)
    recall = TP /(TP+FN)
    accuracy = (TP + TN) /(TP + FP + TN + FN)
    error_rate =(FN + FP)/(TP + FP + TN + FN)
    P, R = precision, recall
    F1_Score = 2*P*R/(P+R)
    return precision, recall, accuracy, error_rate, F1_Score

def specificity_score(true_y, predict_y):
    N = len(true_y)
    TP = sum([1 if (true_y[i]==1 and predict_y[i]==1) else 0 for i in range(N)])
    FP = sum([1 if (true_y[i]==0 and predict_y[i]==1) else 0 for i in range(N)])
    TN = sum([1 if (true_y[i]==0 and predict_y[i]==0) else 0 for i in range(N)])
    FN = sum([1 if (true_y[i]==1 and predict_y[i]==0) else 0 for i in range(N)])
    specificity = TN / (TN+FP)
    return specificity

def get_confusion_matrix(y_true, y_pred):
    df = pd.DataFrame(confusion_matrix(y_true, y_pred), columns = ['pred_N', 'pred_P'])
    df.index = ['true_N','true_P']
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    res = [tn, fp, fn, tp]
    print(f'tn, fp, fn, tp is {res} ')
    print(df)

def get_name_pinyin(name):
    from xpinyin import Pinyin
    p = Pinyin()
    res_list = p.get_pinyin(f'{name}', tone_marks='numbers').split('-')
    res = "".join([s[0] for s in res_list])
    print(res)
    return res 