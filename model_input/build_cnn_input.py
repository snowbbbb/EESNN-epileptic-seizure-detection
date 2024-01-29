
import mne
import mne.annotations
import os,sys 
import numpy as np 
import pandas as pd 
import re, gc
import time 

def reduce_df(df):
    start_mem = round(df.memory_usage().sum() / 1024 ** 3,3)
    for col in tqdm(df.columns):
        df[col] = df[col].astype(np.int8)
    end_mem = round(df.memory_usage().sum() / 1024 ** 3,3)
    print(f'start mem is {start_mem} GB, end mem is {end_mem} GB')    

def look_up_df(path):
    df = pd.read_pickle(path)
    print(f'df shape is {df.shape} ')
    print(df.head())
    return df 

def change_label(label_list):
    print(f'***  change label *** ')
    N = len(label_list)
    pos_ratio = 0.1
    pos_num = int(N*0.1)
    for i in range(pos_num):
        k = np.random.randint(0,N,dtype=int)
        label_list[k] = 1
    return label_list

class CNN_INPUT():
    def __init__(self,config):
        self.config = config 

    def judge_seiz_label(self,tmp_start_time,tmp_end_time,seiz_df):
        label = 0
        for i in range(seiz_df.shape[0]):
            seiz_start_time = seiz_df.at[i,'seiz_start_second']
            seiz_end_time = seiz_df.at[i,'seiz_end_second']
            if tmp_end_time < seiz_start_time or tmp_start_time > seiz_end_time:
                continue
            else:
                label = 1
                return label 
        return label 
 
    def load_raw_data(self,train_patient_ids,train_flag=True):
        #print(f'patient ids are {train_patient_ids}, load raw data, train flag is {train_flag} ')
        config = self.config 
        num_minute = 5  # how many minute data we use for detection 
        num_points = num_minute * 60 *256  # 76800
        window_time = 1
        window_points = window_time * 60 *256
        #print(f'num minute is {num_minute}, num points is  {num_points} ')
        for i,patient_id in enumerate(train_patient_ids):
            raw_path = f'/home/kenan/medic_data/mit_pkl_data/0319/{patient_id}.pkl'
            raw_df = pd.read_pickle(raw_path).reset_index(drop=True) #.iloc[:30000000,:]
            #print(f'df0 columns are {raw_df_0.columns} ')
            #raw_df_0.drop(['time','T8-P8-1'],axis=1,inplace=True)
            #raw_df = (raw_df_0 - raw_df_0.mean()) / (raw_df_0.std()) 
            #print(f'df columns are {raw_df.columns} ')
            #print(f'raw df shape is {raw_df.shape}, df size is {round(raw_df.memory_usage().sum()/ (1024**3),2)} ')
            seiz_path = f'/home/kenan/medic_data/mit_pkl_data/0319/{patient_id}_seiz.pkl'
            seiz_df = pd.read_pickle(seiz_path)
            #print(f'seiz df shape is {seiz_df.shape}, seiz columns are {seiz_df.columns} ')

            len_df = raw_df.shape[0]
            start_index = 0
            end_index = num_points
            cnt = 0
            # ndarray 
            train_data_list = []
            label_list = []
            #print(f'data 1, df lenght is {len_df} ')
            use_print = 1
            while end_index < len_df:
                cnt += 1
                tmp_raw_df = raw_df.iloc[start_index:end_index,:]

                tmp_start_time, tmp_end_time, = tmp_raw_df.at[start_index,'time_second'],tmp_raw_df.at[end_index-1,'time_second']

                tmp_label = self.judge_seiz_label(tmp_start_time,tmp_end_time,seiz_df)

                train_data_list.append(tmp_raw_df.drop(['time','T8-P8-1','time_second'],axis=1).values)
                del tmp_raw_df
                gc.collect()

                label_list.append(tmp_label)
                # updata start/end index , how fool you are
                #start_index = end_index 
                start_index = start_index + window_points
                end_index = start_index + num_points

            del raw_df
            gc.collect()

            #label_list = change_label(label_list)
            train_data_array = np.array(train_data_list).astype(np.int16)
            label_array = np.array(label_list)
            pos_cnt = sum(label_list)
            use_print = 1
            if i==0 and use_print:
                print(f'train array shape is {train_data_array.shape}, look up \n  ')   # {train_data_array[:2]}
                print(f'label array shape is {label_array.shape}, look up \n   ')   # {label_array[:2]}
                print(f'label list len is {len(label_list)}, seiz label cnt is {pos_cnt} ')

        #self.train_data = train_data_array
        #self.train_label = label_array 
        #print(f'***** load cnn raw data done *****')
        return train_data_array, label_array


    def build_train_iterator(self):
        train_patient_ids = ["chb01"]
        #train_data = self.train_data 
        #label_data = self.train_label
        train_data, label_data = self.load_raw_data(train_patient_ids,train_flag=True)
        start = 0
        batch_size =  self.config.batch_size
        end = start + batch_size
        
        while end < train_data.shape[0]:
            #print(f'index start is {start}, end is {end} ')
            batch_data = train_data[start:end,:,:]
            batch_label = label_data[start:end]
            yield batch_data, batch_label

            start = end
            end = start + batch_size 

        #print(f' train iterator end ')

    def build_eval_iterator(self):
        eval_patient_ids = ['chb02']

        eval_data, label_data = self.load_raw_data(eval_patient_ids,train_flag=False)
        start = 0
        batch_size =  self.config.batch_size
        end = start + batch_size
        limit = 10000000
        while end < eval_data.shape[0]:
            #print(f'index start is {start}, end is {end} ')
            batch_data = eval_data[start:end,:,:]
            batch_label = label_data[start:end]
            yield batch_data, batch_label

            start = end
            end = start + batch_size 
            if end > limit:
                break 

        print(f' eval iterator end ')
    
class CNN_INPUT_V2(CNN_INPUT):
    def __init__(self,config):
        super(CNN_INPUT_V2,self).__init__()
        self.config = config
        self.train_eval_ratio = 0.8

    def load_raw_data(self,train_patient_ids,train_flag):
        #print(f'patient ids are {train_patient_ids}, load raw data, train flag is {train_flag} ')
        config = self.config 
        num_minute = 5  # how many minute data we use for detection 
        num_points = num_minute * 60 *256  # 76800
        window_time = 1
        window_points = window_time * 60 *256
        #print(f'num minute is {num_minute}, num points is  {num_points} ')
        for i,patient_id in enumerate(train_patient_ids):
            raw_path = f'/home/lintong/medic_data/mit_pkl_data/0319/{patient_id}.pkl'
            raw_df = pd.read_pickle(raw_path).reset_index(drop=True) #.iloc[:30000000,:]
            #print(f'raw df shape is {raw_df.shape}, df size is {round(raw_df.memory_usage().sum()/ (1024**3),2)} ')
            seiz_path = f'/home/lintong/medic_data/mit_pkl_data/0319/{patient_id}_seiz.pkl'
            seiz_df = pd.read_pickle(seiz_path)
            #print(f'seiz df shape is {seiz_df.shape}, seiz columns are {seiz_df.columns} ')

            len_df = raw_df.shape[0]
            start_index = 0
            end_index = num_points
            cnt = 0
            # ndarray 
            train_data_list = []
            label_list = []
            #print(f'data 1, df lenght is {len_df} ')
            use_print = 1
            while end_index < len_df:
                cnt += 1
                tmp_raw_df = raw_df.iloc[start_index:end_index,:]

                tmp_start_time, tmp_end_time, = tmp_raw_df.at[start_index,'time_second'],tmp_raw_df.at[end_index-1,'time_second']

                tmp_label = self.judge_seiz_label(tmp_start_time,tmp_end_time,seiz_df)

                train_data_list.append(tmp_raw_df.drop(['time','T8-P8-1','time_second'],axis=1).values)
                del tmp_raw_df
                gc.collect()

                label_list.append(tmp_label)
                # updata start/end index , how fool you are
                #start_index = end_index 
                start_index = start_index + window_points
                end_index = start_index + num_points

            del raw_df
            gc.collect()

            #label_list = change_label(label_list)
            train_data_array = np.array(train_data_list).astype(np.int8)
            label_array = np.array(label_list)
            pos_cnt = sum(label_list)
            use_print = 1
            if i==0 and use_print:
                print(f'train array shape is {train_data_array.shape}, look up \n  ')   # {train_data_array[:2]}
                print(f'label array shape is {label_array.shape}, look up \n   ')   # {label_array[:2]}
                print(f'label list len is {len(label_list)}, seiz label cnt is {pos_cnt} ')

        self.all_data = train_data_array
        self.all_label = label_array 

        #print(f'***** load cnn raw data done *****')
        #return train_data_array, label_array


    def build_train_iterator(self):
        #train_patient_ids = ["chb01"]
        #train_data = self.train_data 
        #label_data = self.train_label
        train_data, label_data = self.load_raw_data(train_patient_ids,train_flag=True)
        start = 0
        batch_size =  self.config.batch_size
        end = start + batch_size
        
        while end < train_data.shape[0]:
            #print(f'index start is {start}, end is {end} ')
            batch_data = train_data[start:end,:,:]
            batch_label = label_data[start:end]
            yield batch_data, batch_label

            start = end
            end = start + batch_size 

        #print(f' train iterator end ')

    def build_eval_iterator(self):
        eval_patient_ids = ['chb02']

        eval_data, label_data = self.load_raw_data(eval_patient_ids,train_flag=False)
        start = 0
        batch_size =  self.config.batch_size
        end = start + batch_size
        limit = 10000000
        while end < eval_data.shape[0]:
            #print(f'index start is {start}, end is {end} ')
            batch_data = eval_data[start:end,:,:]
            batch_label = label_data[start:end]
            yield batch_data, batch_label

            start = end
            end = start + batch_size 
            if end > limit:
                break 

        print(f' eval iterator end ')
    


if __name__ == '__main__':

    patient_ids = ["chb01"]
    config = {}
    CNN_Data = CNN_INPUT(config)
    CNN_Data.load_raw_data(patient_ids) 
    train_iterator = CNN_Data.build_train_iterator()
    cnt = 0
    for i,(batch_data, batch_label) in enumerate(train_iterator):
        cnt += 1
        if cnt < 2:
            print(f'batch data shape is {batch_data.shape},\n data is {batch_data} ')
            print(f'batch label shape is {batch_label.shape},\n label is {batch_label} ')
        if cnt % 10 == 0:
            print(f'cnt is {cnt}, i is {i} ')
    print(f'final cnt is {cnt} ')

