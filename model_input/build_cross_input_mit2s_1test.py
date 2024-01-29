
#import mne
#import mne.annotations
import os,sys 
import numpy as np 
import pandas as pd 
import re, gc
import time 
import pickle
import tqdm 
import logging
sys.path.append('/home/zpzhang/pku1')
import torch 
from utils import set_cnn_model_parameters,use_cuda
from data_pku.pos_sample_scale import build_scale_together
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, random_split
# global set params
log_root_path = '/data/zzp/log'

config = set_cnn_model_parameters()

seed = config.seed 
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

np.random.seed(seed)

def plot_embedding(data,label,title):
    x_min,x_max=np.min(data,0),np.max(data,0)
    data=(data-x_min)/(x_max-x_min)
    fig=plt.figure()
    ax = Axes3D(fig)
    for i in range(data.shape[0]):
        ax.scatter(data[i,0],data[i,1],data[i,2],marker=',',color=plt.cm.Set1(label[i]))
    for i in range(2):
        print(plt.cm.Set1(i))
    plt.axis('off')
    plt.title(title,fontsize=140)
    plt.savefig("visualization.png")
    return fig

def GaussianGenerate(data,Gsize):  # size:generate data num
    print(data.shape,data.shape[2])#(1883, 100, 19)
    allmean = np.mean(data, axis=0)
    reRank=[(i+Gsize*j ) for i in range(Gsize) for j in range(data.shape[2])]
   # print("reRank",reRank)
    print("allmean:",allmean.shape)
    print("data[0]",data[:,:,0].shape)
    '''
    for i in range(data.shape[2]):
         cov = np.cov(data[:,:,i].T)
        # print("cov:",cov.shape)
         if i==0:
           allcov=[cov]
         else:
           allcov=np.append(allcov, [cov], axis=0)
    print("allcov:",allcov.shape)       
    for j in range(Gsize):
      for i in range(data.shape[2]):
         cov = np.cov(data[:,:,i].T)
        # print("cov:",cov.shape)
         generate_feature = np.random.multivariate_normal(mean=allmean[:,i], cov=allcov[i,:,:])
         if i==0 and j==0:
           generateFeatureAll = [generate_feature]
         else:
           generateFeatureAll = np.append(generateFeatureAll, [generate_feature], axis=0)
    '''
    for i in range(data.shape[2]):
         cov = np.cov(data[:,:,i].T)
        # print("cov:",cov.shape)
         generateFeature = np.random.multivariate_normal(mean=allmean[:,i], cov=cov, size=Gsize)
         print("generateFeature:",generateFeature.shape) 
         if i==0:
           allcov=[cov]
           generateFeatureAll = generateFeature
         else:
           allcov=np.append(allcov, [cov], axis=0)
           generateFeatureAll = np.append(generateFeatureAll, generateFeature, axis=0)
    generateFeatureAll=generateFeatureAll[reRank,:] 
    print("generateFeatureAll:",generateFeatureAll.shape)       
    generateFeatureAll=generateFeatureAll.reshape((Gsize,data.shape[2],data.shape[1]))   
    generateFeatureAll=np.swapaxes(generateFeatureAll, 1, 2)
       
    print("generateFeatureAll:",generateFeatureAll.shape)
    newdata=np.append(data, generateFeatureAll, axis=0)
    print("newdata:",newdata.shape)
    return newdata
    
def multiGaussianGenerate(data,Gsize):  # size:generate data num

    print(data.shape,data.shape[2])#(1883, 100, 19)
    reshapeData=data.reshape(data.shape[0],-1)
    print(reshapeData.shape,reshapeData.shape)#(1883, 100, 19)
    mean = np.mean(reshapeData, axis=0)
    print("mean:",mean.shape) 
    cov = np.cov(reshapeData.T)
    print("allcov:",cov.shape)       
    generateFeatureAll = np.random.multivariate_normal(mean, cov,size=Gsize)
    print("generateFeatureAll:",generateFeatureAll.shape)   
    generateFeatureAll=generateFeatureAll.reshape((Gsize,data.shape[1],data.shape[2]))      
    print("generateFeatureAll:",generateFeatureAll.shape)
    newdata=np.append(data, generateFeatureAll, axis=0)
    print("newdata:",newdata.shape)
    return newdata

def reduce_df(df):
    start_mem = round(df.memory_usage().sum() / 1024 ** 3,3)
    for col in tqdm(df.columns):
        df[col] = df[col].astype(np.int8)
    end_mem = round(df.memory_usage().sum() / 1024 ** 3,3)
    #print(f'start mem is {start_mem} GB, end mem is {end_mem} GB')    

def look_up_df(path):
    df = pd.read_pickle(path)
    print(f'df shape is {df.shape} ')
    print(df.head())
    return df 

def get_train_data(name_list,flag):
    
    this_root_path = os.path.join(f'/data/zzp/mit_pkl_data/1018_single/{flag}_2')
    arr_list = []
    for name in name_list:
        abs_path = os.path.join(this_root_path,f'{name}_{flag}_array.pkl')
        #print(abs_path)
        with open(abs_path, 'rb') as f:
            arr = pickle.load(f)
            arr_list.append(arr)
    all_arr = np.concatenate(arr_list, axis=0)
    print("all_arr:",all_arr.shape)
    return all_arr


def get_train_data_identity(name_list):
    this_root_path = os.path.join(f'/data/zzp/mit_pkl_data/mit2s_personwise/')
    path_list = os.listdir(this_root_path)
    arr_list = []
    all_label = []
    all_softlabel = []
    i = 0;
    #print("path_list",path_list)
    for name in path_list:
        if name not in name_list:
            continue
        name_path = os.path.join(this_root_path, f'{name}')
       # print(name_path)
        for path in os.listdir(name_path):
            abs_path = os.path.join(name_path, path)
            with open(abs_path, 'rb') as f:
                arr = pickle.load(f)
                arr_list.append(arr)
                all_label.append(i * np.ones(arr.shape[0]))
                all_softlabel.append(i + np.random.rand(arr.shape[0]))
                # print("all_label",all_label)
        i += 1
    all_arr = np.concatenate(arr_list, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_softlabel = np.concatenate(all_softlabel, axis=0)
    # print("all_label",all_label)
    # print("all_arr:",type(all_arr),all_arr.shape)
    return all_arr, all_label, all_softlabel

def get_pretrain_data_old(train_data, test_data, pretrain_size, flag):
    print(f'in get {flag} pretrain data, train_data shape is {train_data.shape}, test data shape is {test_data.shape}, pretrain_size is {pretrain_size} ')
    np.random.shuffle(test_data)
    test_split_index = int(test_data.shape[0] * pretrain_size)
    new_train_data = np.concatenate((train_data, test_data[:test_split_index]), axis=0)
    new_test_data = test_data[test_split_index:]
    print(f'after use {flag} pretrain data, new_train_data shape is {new_train_data.shape}, new_test_data shape is {new_test_data.shape} ')
    return new_train_data, new_test_data

def get_pretrain_data(test_data, pretrain_size, flag):
    print(f'in get {flag} pretrain data, test data shape is {test_data.shape}, pretrain_size is {pretrain_size} ')
    np.random.shuffle(test_data)
    test_split_index = int(test_data.shape[0] * pretrain_size)
    new_train_data = test_data[:test_split_index]
    new_test_data = test_data[test_split_index:]
    print(f'after use {flag} pretrain data, new_train_data shape is {new_train_data.shape}, new_test_data shape is {new_test_data.shape} ')
    return new_train_data, new_test_data


class CNN_INPUT_cross_identity():
    def __init__(self, config):
        self.config = config

        self.train_test_ratio = 0.8
        self.train_eval_ratio = 0.8
        self.pos_train_scale = config.use_scale
        self.use_pretrain = 0.2  # config.use_pretrain

        self.train_name_list = ['chb01', 'chb02', 'chb03', 'chb04', 'chb05',
                        'chb06', 'chb07', 'chb08', 'chb09', 'chb11','chb12','chb13','chb14',
                        'chb15','chb16',  'chb17','chb18', 'chb19','chb20','chb21', 'chb22']

        self.test_name_list =['chb23', ]# ['23', ]
    
        self.test_num = len(self.test_name_list)

        self.load_train_data()

    def load_train_data(self):
        train_name_list = self.train_name_list
        print(f'train_name_list is {train_name_list} ')

        train_data, train_label, train_softlabel = get_train_data_identity(train_name_list)
        print("train_data.shape,train_label.shape:", train_data.shape, train_label.shape)
        with torch.no_grad():
            train_ids = TensorDataset(torch.from_numpy(train_data).cuda(), torch.from_numpy(train_label).cuda())
        train_softlabel_ids = TensorDataset(torch.from_numpy(train_data).cuda(),
                                            torch.from_numpy(train_softlabel).cuda())
        # split train/eval data
        train_size = int(len(train_ids) * 0.8)  # 这里train_size是一个长度矢量，并非是比例，我们将训练和测试进行8/2划分
        test_size = len(train_ids) - train_size
        train_data, eval_data = random_split(train_ids, [train_size, test_size])
        train_softlabel_data, eval_softlabel_data = random_split(train_softlabel_ids, [train_size, test_size])

        '''
        all_scale_way_list = ['add','sub','add_sub','mutiply','random'] * 5
        scale_way_list = all_scale_way_list[:self.pos_train_scale]
        if self.pos_train_scale > 0:
            print(f'use scale is {self.pos_train_scale}, scale_way_list is {scale_way_list} ')
            pos_train_data = build_scale_together(pos_train_data, scale_way_list)
        '''
        self.train_len = len(train_data)
        self.train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=self.config.batch_size,
                                                        drop_last=True)
        self.eval_loader = torch.utils.data.DataLoader(eval_data, shuffle=True, batch_size=self.config.batch_size,
                                                       drop_last=True)
        self.softlabel_train_loader = torch.utils.data.DataLoader(train_softlabel_data, shuffle=True,
                                                                  batch_size=self.config.batch_size, drop_last=True)
        self.softlabel_eval_loader = torch.utils.data.DataLoader(eval_softlabel_data, shuffle=True,
                                                                 batch_size=self.config.batch_size, drop_last=True)


class CNN_INPUT_cross_gaussian():
    def __init__(self, config):
        self.config = config 

        self.train_test_ratio = 0.8
        self.train_eval_ratio = 0.8
        self.pos_train_scale = config.use_scale
        self.use_pretrain = 0.2 #config.use_pretrain

        self.train_name_list = ['chb01', 'chb02', 'chb03', 'chb04', 'chb05',
                        'chb06', 'chb07', 'chb08', 'chb09', 'chb11','chb12','chb13','chb14',
                        'chb15','chb16',  'chb17','chb18', 'chb19','chb20','chb21', 'chb22']

        self.test_name_list =['chb23', ]# ['23', ]
        self.test_num = len(self.test_name_list)

        self.load_train_data()

    def load_train_data(self):
        train_name_list = self.train_name_list
        print(f'train_name_list is {train_name_list} ')
        
        pos_train_data = get_train_data(train_name_list, 'pos')
        neg_train_data = get_train_data(train_name_list, 'neg')
        
        pos_train_data= multiGaussianGenerate(pos_train_data,5*pos_train_data.shape[0])#GaussianGenerate#multiGaussianGenerate
        # split train/eval data
        pos_train_data, pos_eval_data = train_test_split(pos_train_data, test_size=0.2, random_state=42)
        neg_train_data, neg_eval_data = train_test_split(neg_train_data, test_size=0.2, random_state=42)
        self.train_len=pos_train_data.shape[0]+neg_train_data.shape[0]
        self.train_loader = self.build_data_iterator(pos_train_data, neg_train_data, 'train')
        self.eval_loader = self.build_data_iterator(pos_eval_data, neg_eval_data, 'eval')
        

    def load_test_data(self, patient_id=0):
        test_name_list = self.test_name_list[patient_id:patient_id+1]
        print(f'test_name_list is {test_name_list} ')
        pos_test_data = get_train_data(test_name_list, 'pos')
        neg_test_data = get_train_data(test_name_list, 'neg')

        all_scale_way_list = ['add','sub','add_sub','mutiply','random'] * 5
        scale_way_list = all_scale_way_list[:self.pos_train_scale]

        pos_fine_tune_data, pos_test_data = get_pretrain_data(pos_test_data, pretrain_size=self.use_pretrain, flag='pos')
        if self.pos_train_scale > 0:
            print(f'use scale is {self.pos_train_scale}, scale_way_list is {scale_way_list} ')
            pos_fine_tune_data = build_scale_together(pos_fine_tune_data, scale_way_list)

        neg_fine_tune_data, neg_test_data = get_pretrain_data(neg_test_data, pretrain_size=self.use_pretrain, flag='neg')

        test_loader = self.build_data_iterator(pos_test_data, neg_test_data, 'test')
        fine_tune_loader = self.build_data_iterator(pos_fine_tune_data, neg_fine_tune_data, 'fine_tune')
        return fine_tune_loader, test_loader

    def build_data_iterator(self, pos_arr, neg_arr, flag):
        print(f'------ {flag} iterator begin ')
        pos_y = np.ones(pos_arr.shape[0])
        neg_y = np.zeros(neg_arr.shape[0])
        
        x_arr = np.concatenate((pos_arr, neg_arr), axis=0)
        y_arr = np.concatenate((pos_y, neg_y), axis=0)
        print(f'build {flag} iterator here, pos num is {pos_arr.shape[0]}, neg_num is {neg_arr.shape[0]} ')

        tensor_dataset = TensorDataset(
            torch.from_numpy(x_arr),
            torch.from_numpy(y_arr)
        )
        shuffle = True
        if flag == 'test':
            shuffle = False
        if flag=='train':
            print("train data pos num:",pos_arr.shape[0])
            print("train data neg num:",neg_arr.shape[0])
        data_loader = DataLoader(dataset=tensor_dataset,
            shuffle=shuffle,
            batch_size = self.config.batch_size,
            drop_last = True
        )
        return data_loader
        
        

if __name__ == '__main__':

    #CNN_Data = CNN_INPUT_cross(config)
    CNN_Data = CNN_INPUT_cross_fine_tune(config)
    CNN_Data.load_test_data()






