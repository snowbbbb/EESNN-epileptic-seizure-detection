"""
this code works,
1. 36 recurrent neurons will lead to 84.4 percent on testset
2. 24 recurrent neurons will lead to 81.8 percent on testset
"""

import os, sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import numpy as np
import time
import copy
from torch.autograd import Variable
import pandas as pd
from torch.optim.lr_scheduler import StepLR,MultiStepLR
import matplotlib.pyplot as plt
import math
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
from sklearn.metrics import confusion_matrix
import scipy.io
from model_input.build_cross_input_beiyi2s_noFT import CNN_INPUT_cross, CNN_INPUT_cross_fine_tune, \
    CNN_INPUT_cross_gaussian
# from model_input.build_cross_input_mit5s_noFT import CNN_INPUT_noFT, CNN_INPUT_cross, CNN_INPUT_cross_fine_tune,CNN_INPUT_cross_gaussian
# from model_input.build_cross_input_beiyi5s_noFT import CNN_INPUT_noFT, CNN_INPUT_cross, CNN_INPUT_cross_fine_tune
from utils import set_cnn_model_parameters, use_cuda, evaluate_result
from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score
from tools.logger import Logger
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from utils import specificity_score, get_confusion_matrix

params = set_cnn_model_parameters()
CNN_Data = CNN_INPUT_cross_gaussian(params)  # CNN_INPUT_noFT(params)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device =  torch.device("cpu")
print('device: ',device)

'''
STEP 3a_v2: CREATE Adaptative spike MODEL CLASS
'''
b_j0 = 0.01  # neural threshold baseline
tau_m = 20  # ms membrane potential constant
R_m = 1  # membrane resistance
dt = 1  #
gamma = .5  # gradient scale
lens = 0.5

surrograte_type = 'MG'
print('surrograte_type: ',surrograte_type)

neuron_type = 'adaptive'
print('neuron_type: ',neuron_type)


def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma

# define approximate firing function

class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # temp = abs(input) < lens
        scale = 6.0
        hight = .15
        if surrograte_type == 'G':
            temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        elif surrograte_type == 'MG':
            temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
                - gaussian(input, mu=lens, sigma=scale * lens) * hight \
                - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        elif surrograte_type =='linear':
            temp = F.relu(1-input.abs())
        elif surrograte_type == 'slayer':
            temp = torch.exp(-5*input.abs())
        return grad_input * temp.float() * gamma
# membrane potential update

act_fun_adp = ActFun_adp.apply

if neuron_type =='adaptive':
    def mem_update_adp(inputs, mem, spike, tau_m,tau_adp, b, isAdapt=1, dt=1):
        alpha = torch.exp(-1. * dt / tau_m).cuda()
        ro = torch.exp(-1. * dt / tau_adp).cuda()
        # tau_adp is tau_adaptative which is learnable # add requiregredients
        if isAdapt:
            beta = 1.8
        else:
            beta = 0.

        b = ro * b + (1 - ro) * spike
        B = b_j0 + beta * b

        mem = mem * alpha + (1 - alpha) * R_m * inputs - B * spike * dt
        inputs_ = mem - B
        spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
        return mem, spike, B, b

if neuron_type  == 'LIF':
    def mem_update_adp(inputs, mem, spike, tau_m, tau_adp, b, dt=1, isAdapt=1):

        b = 0
        B = .5
        alpha = torch.exp(-1. * dt / tau_adp).cuda()
        mem = mem *.618 + inputs#*(1- alpha)
        inputs_ = mem - B
        spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
        mem = (1-spike)*mem 
        return mem, spike, B, b

class RNN_s(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, sub_seq_length,criterion):
        super(RNN_s, self).__init__()
        self.criterion = criterion

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sub_seq_length = sub_seq_length
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

        self.tau_adp_h = nn.Parameter(torch.Tensor(hidden_size))
        self.tau_adp_o = nn.Parameter(torch.Tensor(output_size))
        
        self.tau_m_h = nn.Parameter(torch.Tensor(hidden_size))
        self.tau_m_o = nn.Parameter(torch.Tensor(output_size))
        
        nn.init.orthogonal_(self.h2h.weight)
        nn.init.xavier_uniform_(self.i2h.weight)
        nn.init.xavier_uniform_(self.h2o.weight)
        nn.init.constant_(self.i2h.bias, 0)
        nn.init.constant_(self.h2h.bias, 0)
        nn.init.constant_(self.h2o.bias, 0)

        nn.init.constant_(self.tau_adp_h, 7) #7
        nn.init.constant_(self.tau_adp_o, 100)
        nn.init.constant_(self.tau_m_h, 20) #7
        nn.init.constant_(self.tau_m_o, 20)
        
        self.b_h = self.b_o = 0

    def forward(self, input,labels):
        self.b_h = self.b_o = b_j0
        total_spikes = 0
        # Feed in the whole sequence
        batch_size, seq_num, input_dim = input.shape
        hidden_mem = hidden_spike = (torch.rand(batch_size, self.hidden_size)*b_j0).cuda()
        output_mem = output_spike = out_spike = (torch.rand(batch_size, self.output_size)*b_j0).cuda()
        output_spike_sum = torch.zeros(batch_size,seq_num, self.output_size).cuda()
        self.b_h = self.b_o = 0.01

        max_iters = 100
        loss = 0

        output_ = []
        I_h = []
        predictions = []
        for i in range(max_iters): # Go through the sequence
            if i < seq_num:
                input_x = input[:, i, :]
            else:
                input_x = torch.zeros(batch_size,input_dim)

            #################   update states  #########################
            input_x=input_x.to(device)
            #print("type",input_x.device,hidden_spike.device)
            h_input = self.i2h(input_x.float()) + self.h2h(hidden_spike)
            hidden_mem, hidden_spike, theta_h, self.b_h = mem_update_adp(h_input,hidden_mem, hidden_spike,self.tau_m_h,
                                                                            self.tau_adp_h, self.b_h,isAdapt=0)#0

            I_h.append(h_input.data.cpu().numpy())
            o_input = self.h2o(hidden_spike)
            output_mem, output_spike, theta_o, self.b_o = mem_update_adp(o_input,output_mem,output_spike,self.tau_m_o, 
                                                                         self.tau_adp_o, self.b_o,isAdapt=1)#, dt=input_dt_o)
            output_spike_sum[:,i,:] = output_spike
            total_spikes = total_spikes + int(hidden_spike.sum() + output_spike.sum())
            #################   classification  #########################
            if i >= self.sub_seq_length:
                output_sumspike = output_mem #output_spike_sum[:, i-1:i, :].sum(axis=1)
                output_sumspike = F.log_softmax(output_sumspike,dim=1)

                predictions.append(output_sumspike.data.cpu().numpy())
                output_.append(output_sumspike.data.cpu().numpy())
                loss += self.criterion(output_sumspike, labels)

        predictions = torch.tensor(predictions)
        return predictions, loss , total_spikes

    def predict(self,input, lablel):
        prediction = self.forward(input, lablel)
        # prediction, _, total_spikes = self.forward(dt_h, dt_o, max_i, input, lablel)
        return prediction


def train(model,dims, loader,optimizer,scheduler=None,num_epochs=10):
    best_acc = 0
    path = '/data/zzp/result/model_save/mit/'  # .pth'
    acc_list=[]
    for epoch in range(num_epochs):
        train_acc = 0
        train_loss_sum = 0
        sum_samples = 0
        for i, (images, labels) in enumerate(loader):
            #print("images",type(images),images.shape)
            images = images.requires_grad_().to(device)
            labels = labels.long().to(device)
            #images = images.view(-1, dims["samples"], dims["input"]).requires_grad_().to(device)#images.view(-1, num_samples, input_dim).requires_grad_().to(device)
            #labels = labels.view((-1,dims["samples"])).long().to(device)
            #print("labels",type(labels),labels.shape)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output/logits
            predictions, train_loss,_ = model(images, labels)
            #print("predicted0",predictions.shape,predictions)
            predictions = predictions.mean(0)
            #print("predicted1",predictions.shape)


            _, predicted = torch.max(predictions.data, 1)
            #print("predicted2",predicted.shape,predicted)
            # Getting gradients w.r.t. parameters
            train_loss.backward()
            train_loss_sum += train_loss
            optimizer.step()

            labels = labels.cpu()
            #print(predicted.type(), labels.type())
            predicted = predicted.cpu().t()
            
            train_acc += (predicted == labels).sum()
            #sum_samples = sum_samples + predicted.numel()
            # train_acc += (predicted == labels).sum()
        if scheduler is not None:
            scheduler.step()
        train_acc = train_acc.data.cpu().numpy() / 18099
        if train_acc>best_acc and train_acc>0.80:
            best_acc = train_acc
            torch.save(model.state_dict(), path+str(best_acc)[:5]+'-wt-new.pth')

        acc_list.append(train_acc)
        print('epoch: {:3d}, Train Loss: {:.4f}, Train Acc: {:.4f}'.format(epoch,
                                                                           train_loss_sum.item()/len(loader)/(1300-dims["sub_seq"]),
                                                                           train_acc), flush=True)
    return acc_list
import scipy.signal as ssg
def convert_seq(x,threshold=0.03):
    l = len(x)
    x= ssg.savgol_filter(x, 5, 3)
    X = np.zeros((l,2))
    for i in range(len(x)-1):
        if x[i+1] - x[i] >= threshold:
            X[i,0] = 1
        elif x[i] - x[i+1] >= threshold:
            X[i,1] = 1
    return X


def expand_dim(x, N):
    y = np.zeros((x.shape[0], x.shape[1], N))
    for i in range(x.shape[0]):
        y[i, :, :] = np.tile(x[i,:], (N,1)).transpose()

    return y

def lbl_to_spike(prediction):
    N = len(prediction)
    detections = np.zeros(N)
    for i in range(1, N):
        if (prediction[i] != prediction[i-1]):
            detections[i] = prediction[i]+1
    return detections


def calculate_stats(prediction, lbl, tol):
    decisions = lbl_to_spike(prediction)
    labs = lbl_to_spike(lbl)

    lbl_indices = np.nonzero(labs)
    lbl_indices = np.array(lbl_indices).flatten()

    dist = np.zeros((len(lbl_indices), 6))
    for i in range(len(lbl_indices)):
        index = lbl_indices[i]
        lab = int(labs[index])
        dec_indices = np.array(np.nonzero((decisions-lab) == 0)).flatten()  #indices where decisions == lab
        if len(dec_indices) == 0:
            dist[i, lab - 1] = 250
            continue
        j = np.argmin(np.abs(dec_indices - index))  # j is closest val in dec_indices to index
        dist[i, lab-1] = abs(dec_indices[j]-index)
        if (dist[i, lab-1] <= tol):
            decisions[dec_indices[j]] = 0 # mark as handled

    mean_error = np.mean(dist, axis=0)
    TP = np.sum(dist <= tol, axis=0)
    FN = np.sum(dist > tol, axis=0)

    FP = np.zeros(6)
    for i in decisions[(decisions > 0)]:
        FP[int(i-1)] += 1

    return mean_error, TP, FN, FP

def convert_dataset_wtime(mat_data):
    X = mat_data["x"]
    Y = mat_data["y"]
    t = mat_data["t"]
    Y = np.argmax(Y[:, :, :], axis=-1)
    d1,d2 =  t.shape

    # dt = np.zeros((size(t[:, 1]), size(t[1, :])))
    dt = np.zeros((d1,d2))
    for trace in range(d1):
        dt[trace, 0] = 1
        dt[trace, 1:] = t[trace, 1:] - t[trace, :-1]

    return dt, X, Y


def load_max_i(mat_data):
    max_i = mat_data["max_i"]
    return np.array(max_i.squeeze(),dtype=np.float16)
start_time_tag = params.time
if len(start_time_tag) < 5:
    start_time_tag = time.strftime("%m%d%H%M")
model_root_path = '/data/zzp/result/model_save/mit/'
if not os.path.exists(model_root_path):
    os.makedirs(model_root_path)

log_root_path = '/data/zzp/log'
dataset = params.dataset
model_name = 'pretrain_cnn'
log_path = os.path.join(log_root_path, f'{dataset}_{model_name}_tag{start_time_tag}.log')
log = Logger(log_path)
log.logger.info(f'log path is {log_path} ')

EPOCHS = params.epochs
BATCH_SIZE = params.batch_size  # 32 #params.batch_size
FREQUENCY = params.frequency

# use_cw_block = params.use_cw_block

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_id = params.model_id

model_name = params.model_name

log.logger.info(f'ResNet18 model version is  v{model_id} ')
log.logger.info(f'params are \n {params} ')
# model params

loss_function = nn.BCELoss()  # F.mse_loss#

# get data
# CNN_Data = CNN_INPUT_cross(params)

time_window_size = params.time_window_size  #


def train_once(model, optimizer, train_iterator, params, epoch_i):
    # print(f'-- epoch is {epoch_i}, cnn model train begin -- ')
    model.train()
    train_loss_sum  = 0.0
    label_list = []
    predict_list = []
    model.use_print = params.use_print
    mean_loss_list = [0]

    pbar = tqdm(total=len(train_iterator))
    for i, (batch_x, batch_y) in enumerate(train_iterator):
        # print(f'test 0, batch x shape is {batch_x.shape}, batch y shape is {batch_y.shape}  ')
        batch_x = batch_x[:, :, :-1]  # [batch_size,128,19-1]
        batch_x = batch_x.requires_grad_().to(device)
        batch_y = batch_y.long().to(device)
        optimizer.zero_grad()
        # Forward pass to get output/logits
        predictions, train_loss, _ = model(batch_x, batch_y)
        # print("predicted0",predictions.shape,predictions)
        predictions = predictions.mean(0)
        _, predicted = torch.max(predictions.data, 1)

        #print(train_loss)
        if i == 0:
            pass
            # print(f'i is {i}, loss is {loss.item()},\n predict_y is {predict_y[:10]} \n batch_y is {batch_y[:10]} ')
        train_loss.backward()
        train_loss_sum += train_loss.item()
        optimizer.step()

        label_list.extend(batch_y.reshape(-1).tolist())
        predict_list.extend(predicted.data.cpu().tolist())
        use_print = 1
    
    # print("predict_list:",predict_list)
    predict_label = [1 if p > params.threshold else 0 for p in predict_list]
    # check_result = [(label_list[i],predict_label[i]) for i in range(100)]
    accuracy = round(accuracy_score(label_list, predict_label), 4)
    precision = round(precision_score(label_list, predict_label), 4)
    recall = round(recall_score(label_list, predict_label), 4)
    F1_score = round(f1_score(label_list, predict_label), 4)
    train_auc = round(roc_auc_score(label_list, predict_list), 4)
    specificity = round(specificity_score(label_list, predict_label), 4)
    # log.logger.info(f'epoch {epoch_i} end, train auc is {train_auc} ,train_loss is {round(train_loss,4)} ')
    # log.logger.info(f'accuracy is {accuracy}, --- train recall is {recall} --- , f1 score is {F1_score} ')

    return accuracy, precision, recall, specificity, train_auc, train_loss_sum, model


def evaluate(model, eval_iterator, params, epoch_i, run_mode='eval'):
    log.logger.info(f'-- epoch is {epoch_i}, run_mode is {run_mode}, begin -- ')
    model.eval()
    eval_loss_sum = 0.0
    label_list = []
    predict_list = []
    for i, (batch_x, batch_y) in enumerate(eval_iterator):
        batch_x = batch_x[:, :, :-1]  # [batch_size,128,19-1]
        batch_x = batch_x.to(device)
        batch_y = batch_y.long().to(device)
        # Forward pass to get output/logits
        predictions, eval_loss, _ = model(batch_x, batch_y)
        # print("predicted0",predictions.shape,predictions)
        predictions = predictions.mean(0)
        _, predicted = torch.max(predictions.data, 1)
        eval_loss_sum += eval_loss.item()


        label_list.extend(batch_y.reshape(-1).tolist())
        predict_list.extend(predicted.data.cpu().tolist())
    predict_label = [1 if p > params.threshold else 0 for p in predict_list]
    # check_result = [(label_list[i],predict_label[i]) for i in range(100)]
    accuracy = round(accuracy_score(label_list, predict_label), 4)
    precision = round(precision_score(label_list, predict_label), 4)
    recall = round(recall_score(label_list, predict_label), 4)
    F1_score = round(f1_score(label_list, predict_label), 4)
    eval_auc = round(roc_auc_score(label_list, predict_list), 4)
    specificity = round(specificity_score(label_list, predict_label), 4)
    log.logger.info(
        f'run mode is {run_mode}, epoch i is {epoch_i}, {run_mode} auc is {eval_auc} , {run_mode} loss is {eval_loss} ')
    log.logger.info(
        f'accuracy is {accuracy}, --- {run_mode} recall is {recall} ---, f1 score is {F1_score}, show confusion matrix below ')
    # get_confusion_matrix(label_list, predict_label)

    return accuracy, precision, recall, specificity, eval_auc, eval_loss_sum


def run_train(model, optimizer, scheduler=None,use_eval=1):
    start_time = time.time()
    log.logger.info(f'run begin, start time is {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} ')
    start_time_1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    train_auc_list = []
    eval_auc_list = []
    # sensitivity, specificity
    train_result_df = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'specificity', 'AUC', 'loss'])
    eval_result_df = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'specificity', 'AUC', 'loss'])

    best_model = model
    best_score = 0
    best_epoch = 0

    log.logger.info(f'total epochs is {EPOCHS} ')

    CNN_Data.load_train_data()
    train_iterator = CNN_Data.train_loader
    eval_iterator = CNN_Data.eval_loader
    for i in range(EPOCHS):
        # model,optimizer,train_iterator,params,epoch_i
        accuracy, precision, recall, specificity, train_auc, train_loss, tmp_model = train_once(model, optimizer,
                                                                                                train_iterator, params,
                                                                                                i)
        if scheduler is not None:
          scheduler.step()
        train_auc_list.append(train_auc)
        train_result_df.loc[i] = [accuracy, precision, recall, specificity, train_auc, train_loss]
        if use_eval:
            accuracy, precision, recall, specificity, eval_auc, eval_loss = evaluate(model, eval_iterator, params, i)
            eval_auc_list.append(eval_auc)
            #print("type",type(accuracy),type(precision),type(recall),type(specificity),type(eval_auc),type(eval_loss))
            eval_result_df.loc[i] = [accuracy, precision, recall, specificity, eval_auc, eval_loss]

            # all_score = np.sum([accuracy,precision,recall*2,specificity,eval_auc])
            all_score = recall * 1.5 + specificity + eval_auc
            if all_score > best_score:
                model_path = os.path.join(model_root_path, f'pretrain_bestmodel.pkl')  # _{start_time_tag}.pkl')
                torch.save(model, model_path)
                # print(f'improved, save epoch {i} model to {model_path} ')
                best_score = all_score
                best_epoch = i
                # best_model = copy.deepcopy(tmp_model)
                best_model.best_epoch = i
    best_model = torch.load(f'/data/zzp/result/model_save/mit/pretrain_bestmodel.pkl')  # _{start_time_tag}.pkl')
    torch.cuda.empty_cache()
    end_time = time.time()
    if use_eval == 0:
        return best_model
    # log.logger.info(f'run end, start time is {start_time_1 } ')
    # log.logger.info(f'run end, end time is {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) } ')
    print(f'all run time is {round((end_time - start_time) / 60, 2)} min ')
    print(f'train_auc list is \n {train_auc_list} ')
    print(f'eval_auc list is \n {eval_auc_list} ')
    print(f'train result df is ')
    print(train_result_df.tail(10))
    print(f'eval result df is ')
    print(eval_result_df.tail(20))
    log.logger.info(f'----train-eval best score is {best_score}, best_epoch is {best_epoch} ----')
    # print(f'test result dic is :\n {test_result_dic} ')
    use_scale = params.use_scale
    best_model_save_path = os.path.join(model_root_path,
                                        f'resnet_scale{use_scale}_epoch{best_epoch}of{EPOCHS}_{start_time_tag}.pkl')
    torch.save(best_model, best_model_save_path)
    log.logger.info(f'-------------- best_pretrain_model_path save to : \n {best_model_save_path} ')
    return best_model_save_path


from utils import get_name_pinyin

'''
def run_test_once(best_model):
    optimizer = torch.optim.Adam(best_model.parameters(),lr=params.lr)
    test_num = CNN_Data.test_num 
    test_result_df = pd.DataFrame(columns=['patient','Sensitivity','Specificity','Accuracy','AUC']) 
    for i in range(test_num):
        fine_tune_loader, test_loader = CNN_Data.load_test_data(patient_id=i)
        patient_name = CNN_Data.test_name_list[i]
        name_py = get_name_pinyin(patient_name)
        print(f'i is {i}, name is {patient_name}, py is {name_py} ')
        #print(f'train best model.best_epoch is {best_model.best_epoch}, tmp model epoch is {tmp_model.best_epoch} ')
        test_result_dic = {}
        test_accuracy,test_precision,test_sensitivity,test_specificity,test_auc, _ = evaluate(best_model,test_loader,params,run_mode='test',epoch_i=i)
        test_result_df.loc[i] = [name_py, test_sensitivity, test_specificity, test_accuracy, test_auc]

    mean_result_list = ['mean']
    for col in test_result_df.columns[1:]:
        #print(f'col is {col}')
        m = test_result_df[col].mean()
        mean_result_list.append(m)
    test_result_df.loc[test_num] = mean_result_list

    #print(f'show test result, use_scale is {params.use_scale}, use pretrain is {params.use_pretrain} ')
    #print(test_result_df)

    #print(f'run test end')
    return test_result_df
'''


def run_test_once(best_model):
    # optimizer = torch.optim.Adam(best_model.parameters(),lr=params.lr)
    test_num = CNN_Data.test_num
    test_result_df = pd.DataFrame(columns=['patient', 'Sensitivity', 'Specificity', 'Accuracy', 'AUC'])
    with torch.no_grad():
        for i in range(test_num):
            test_loader = CNN_Data.load_test_data(patient_id=i)
            patient_name = CNN_Data.test_name_list[i]
            name_py = get_name_pinyin(patient_name)
            print(f'i is {i}, name is {patient_name}, py is {name_py} ')
            # print(f'train best model.best_epoch is {best_model.best_epoch}, tmp model epoch is {tmp_model.best_epoch} ')
            test_result_dic = {}
            test_accuracy, test_precision, test_sensitivity, test_specificity, test_auc, _ = evaluate(best_model,
                                                                                                      test_loader,
                                                                                                      params,
                                                                                                      run_mode='test',
                                                                                                      epoch_i=i)
            test_result_df.loc[i] = [name_py, test_sensitivity, test_specificity, test_accuracy, test_auc]

        mean_result_list = ['mean']
        for col in test_result_df.columns[1:]:
            # print(f'col is {col}')
            m = test_result_df[col].mean()
            mean_result_list.append(m)
        test_result_df.loc[test_num] = mean_result_list

    # print(f'show test result, use_scale is {params.use_scale}, use pretrain is {params.use_pretrain} ')
    # print(test_result_df)

    # print(f'run test end')
    return test_result_df


def run_kfold_test(best_pretrain_model_path):
    log.logger.info(f'run kfold test begin, best model path is {best_pretrain_model_path} ')
    best_model = torch.load(best_pretrain_model_path)
    # optimizer = torch.optim.Adam(best_model.parameters(),lr=0.001)
    optimizer = torch.optim.Adam(best_model.parameters(), lr=params.lr)
    test_num = CNN_Data.test_num
    kfold_test_result_df = pd.DataFrame(columns=['patient', 'Sensitivity', 'Specificity', 'Accuracy', 'AUC'])
    fold_n = params.fold_n
    log.logger.info(f'run kfold test begin, fold n is {fold_n} ')
    for i in range(fold_n):
        base_model = copy.deepcopy(best_model)
        result_df = run_test_once(base_model)
        kfold_test_result_df = pd.concat([kfold_test_result_df, result_df])
        print(kfold_test_result_df.shape)

    # path = '/mnt/data1/user/kenan/result/kfold_result_df.pkl'
    # kfold_test_result_df.to_pickle(path)
    # print(kfold_test_result_df)
    final_result_df = merge_result(kfold_test_result_df)
    print(f'show final result df ')
    print(final_result_df)


def merge_result(kfold_df):
    # path = '/mnt/data1/user/kenan/result/kfold_result_df.pkl'
    # kfold_df = pd.read_pickle(path)
    name_list = list(kfold_df['patient'].unique())
    # print(name_list)
    # print(kfold_df)
    # for name in name_list:
    df = kfold_df.groupby(['patient']).mean()

    new_df = pd.DataFrame(columns=['patient', 'Sensitivity', 'Specificity', 'Accuracy', 'AUC'])
    for i, index in enumerate(list(df.index)):
        if index == 'mean':
            continue
        new_df.loc[i] = [index] + list(df.loc[index].values)

    # print(new_df)
    # print(df)
    test_num = len(list(new_df.index))
    mean_result_list = ['mean']
    for col in new_df.columns[1:]:
        # print(f'col is {col}')
        m = new_df[col].mean()
        mean_result_list.append(m)
    new_df.loc[test_num] = mean_result_list
    # print(new_df)
    return new_df


if __name__ == '__main__':
    cuda_id = params.cuda

    with torch.cuda.device(cuda_id):
        batch_size = 64
        n_iters = 300000
        lens = 0.5  # hyper-parameters of approximate function
        # num_epochs = 1#250  # n_iters / (len(train_dataset) / batch_size)
        num_epochs = 40  # 400 #400
        # nb_of_batch = nb_of_sample // batch_size

        sub_seq_length = 10
        nb_of_sample, seq_dim, input_dim = 18099,100,18
        # L = seq_dim - sub_seq_length
        hidden_dim = 36
        if neuron_type == 'LIF': hidden_dim = 36
        output_dim = 2

        criterion = nn.NLLLoss()
        model = RNN_s(input_size=input_dim, hidden_size=hidden_dim,
                      output_size=output_dim, sub_seq_length=sub_seq_length, criterion=criterion)

        model.to(device)

        learning_rate = params.lr#1e-2

        base_params = [model.i2h.weight, model.i2h.bias, model.h2h.weight,
                       model.h2h.bias, model.h2o.weight, model.h2o.bias]

        optimizer = torch.optim.Adam([
            {'params': base_params},
            {'params': model.tau_m_h, 'lr': learning_rate * 3},
            {'params': model.tau_m_o, 'lr': learning_rate * 2},
            {'params': model.tau_adp_h, 'lr': learning_rate * 3},
            {'params': model.tau_adp_o, 'lr': learning_rate * 2}, ],
            lr=learning_rate)
        # scheduler = StepLR(optimizer, step_size=100, gamma=.75) # gaussian
        scheduler = StepLR(optimizer, step_size=100, gamma=.5)  # LIF

        # training network
        dims = {
            "samples": seq_dim,
            "input": input_dim,
            "hidden": hidden_dim,
            "output": output_dim,
            "sub_seq": sub_seq_length
        }

        best_pretrain_model_path = run_train(model, optimizer=optimizer, scheduler=scheduler, use_eval=1)
        # best_model = torch.load(best_pretrain_model_path)
        run_kfold_test(best_pretrain_model_path)
        log.logger.info(f'-------------- best_pretrain_model_path is : \n {best_pretrain_model_path} ')
        # run_test_once(best_model)
        # run_kfold_test(best_model)
    # merge_result()
    end_time_tag = time.strftime("%m%d%H%M")
    log.logger.info(f'start time tag is {start_time_tag}, end_time_tag is {end_time_tag} ')
    log.logger.info(f'log path is {log_path} ')






