import os,sys 
import numpy as np 
import pandas as pd 
import pickle
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import time
import copy 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from utils import specificity_score, get_confusion_matrix
sys.path.append('/home/zpzhang/pku1')

from tools.logger import Logger
from model_input.build_cross_input_mit4s_noFT import CNN_INPUT_cross_gaussian,CNN_INPUT_cross_identity#from model_input.build_cross_input_mit5s_noFT import CNN_INPUT_noFT, CNN_INPUT_cross, CNN_INPUT_cross_fine_tune,CNN_INPUT_cross_gaussian
#from model_input.build_cross_input_beiyi5s_noFT import CNN_INPUT_noFT, CNN_INPUT_cross, CNN_INPUT_cross_fine_tune
from utils import set_cnn_model_parameters,use_cuda, evaluate_result
#from models.model_cw_srnet import cw_srnet
from spikingjelly.clock_driven import neuron, surrogate, functional
#from spikingjelly.clock_driven.model import train_imagenet, spiking_resnet
#from models.snn_ide_fc import SNNIDEFCNet
from models.snn_ide_conv_multilayer_DANN_MIT import SNNIDEConvMultiLayerNet
#from models.snn_ide_conv_multilayer import SNNIDEConvMultiLayerNet
torch.set_num_threads(4)
params = set_cnn_model_parameters()
time_window_size = params.time_window_size

#CNN_Data = CNN_INPUT_cross_fine_tune(params)
CNN_Data = CNN_INPUT_cross_gaussian(params) #CNN_INPUT_noFT(params)
CNN_Data_identity = CNN_INPUT_cross_identity(params)

# global set params

#start_time_tag = time.strftime("%m%d%H%M%S")
start_time_tag = params.time 
if len(start_time_tag)<5:
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
print('log_path:',log_path)
EPOCHS = params.epochs
BATCH_SIZE = params.batch_size  #32 #params.batch_size
FREQUENCY = params.frequency 

#use_cw_block = params.use_cw_block

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_id = params.model_id

model_name = params.model_name 


log.logger.info(f'ResNet18 model version is  v{model_id} ')
log.logger.info(f'params are \n {params} ')
# model params

loss_function =  nn.BCELoss()#F.mse_loss#
loss_funct_identity= nn.CrossEntropyLoss()
# get data 
#CNN_Data = CNN_INPUT_cross(params)

time_window_size = params.time_window_size # 

def train_once(model,optimizer,train_iterator,params,epoch_i):
    #print(f'-- epoch is {epoch_i}, cnn model train begin -- ')
    model.train()
    train_loss = 0.0
    label_list = []
    predict_list = []
    model.use_print = params.use_print
    mean_loss_list = [0]

    pbar = tqdm(total=len(train_iterator))
    for i,(batch_x,batch_y) in enumerate(train_iterator):
        #print(f'test 0, batch x shape is {batch_x.shape}, batch y shape is {batch_y.shape}  ')
        batch_x = batch_x[:,:,:-1] # [batch_size,128,19-1]
        batch_x = use_cuda(batch_x.float()) 
        #print("batch_x.shape:",batch_x.shape,batch_x)
        batch_y = use_cuda(batch_y.float())
        model_input = batch_x.view(BATCH_SIZE,1,FREQUENCY*time_window_size,-1)
        #print("here! model_input.shape:",model_input.shape)
        #model_input=model_input.view(BATCH_SIZE, 1,time_window_size,FREQUENCY,-1)# [batchsize, 1, 2,50, 18]
        #model_input=model_input.permute(2,0,4,3,1)

        #print("here! model_input.shape:",model_input.shape)
        #print(f'test 0.5')
        predict_y = model(model_input)
        #print("predict_y.shape:",predict_y.shape)
        #predict_y =predict_y.mean(0)
        #print("new predict_y.shape:",predict_y.shape)
        out = torch.sigmoid(predict_y)
        predict_y, _ = torch.max(out,dim=1)
        #print("probability.shape:",probability.shape)
        #print("batch_y.shape:",batch_y.shape,batch_y)
        #label_onehot = F.one_hot(batch_y.to(torch.int64), 2).float().detach()
        #print("label_onehot.shape:",label_onehot.shape)#,label_onehot)
        loss = loss_function(predict_y,batch_y) 
        #loss.requires_grad_()
        print(loss)
        if i == 0:
            pass
            #print(f'i is {i}, loss is {loss.item()},\n predict_y is {predict_y[:10]} \n batch_y is {batch_y[:10]} ')

        optimizer.zero_grad() # zero gradient
        loss.backward()
        optimizer.step()    # update gradient 

        functional.reset_net(model)
        

        train_loss += loss.item()
        if loss.item() > -100 and loss.item() < 1e8:
            mean_loss_list.append(loss.item())
            if len(mean_loss_list)>1000:
                mean_loss_list.clear()
        mean_loss = np.mean(mean_loss_list)
        pbar.set_postfix({'mean_loss':mean_loss})
        pbar.update(1)

        label_list.extend(batch_y.reshape(-1).tolist())
        predict_list.extend(predict_y.data.cpu().tolist())
        use_print = 1
    #print("predict_list:",predict_list)
    predict_label = [1 if p>params.threshold else 0 for p in predict_list]
    #check_result = [(label_list[i],predict_label[i]) for i in range(100)]
    accuracy = round(accuracy_score(label_list,predict_label),4)
    precision = round(precision_score(label_list,predict_label),4)
    recall = round(recall_score(label_list,predict_label),4)
    F1_score = round(f1_score(label_list,predict_label),4)
    train_auc = round(roc_auc_score(label_list,predict_list),4)
    specificity = round(specificity_score(label_list,predict_label),4)
    #log.logger.info(f'epoch {epoch_i} end, train auc is {train_auc} ,train_loss is {round(train_loss,4)} ')
    #log.logger.info(f'accuracy is {accuracy}, --- train recall is {recall} --- , f1 score is {F1_score} ')

    return accuracy,precision,recall,specificity,train_auc, train_loss, model 
    
def train_once_DANN(model,optimizer,optimizer2,train_iterator,train_identity_dataloader,len_train_iter,params,epoch_i):
    #print(f'-- epoch is {epoch_i}, cnn model train begin -- ')
    model.train()
    train_loss = 0.0
    label_list = []
    predict_list = []
    model.use_print = params.use_print
    mean_loss_list = [0]
    train_identity_iterator = iter(train_identity_dataloader)
    eval_acc=0

    #pbar = tqdm(total=len(train_iterator))
    for i in range(len_train_iter):
        #print(f'test 0, batch x shape is {batch_x.shape}, batch y shape is {batch_y.shape}  ')
        '''
        #if len_train_iter is the max of two length, the below is needed,and zhuxiao the more below "data_identity=train_identity_iterator.next()" 
        try:
          data_identity=train_identity_iterator.next()
        except StopIteration:
          train_identity_iterator = iter(train_identity_dataloader)
          data_identity=train_identity_iterator.next()
        '''
        data_identity=train_identity_iterator.next()
        #print("i:",i)
        batch_x_iden, batch_y_iden = data_identity
        batch_x_iden = batch_x_iden[:,:,:-1] # [batch_size,128,19-1]
        batch_x_iden = use_cuda(batch_x_iden.float()) 
        batch_y_iden = use_cuda(batch_y_iden.float())
        #print("batch_x_iden:",batch_x_iden.shape,FREQUENCY,time_window_size)
        model_input_iden = batch_x_iden.view(BATCH_SIZE,1,FREQUENCY*time_window_size,-1)
        #print("model_input_iden:",model_input_iden.shape,FREQUENCY,time_window_size)
        predict_y_iden = model(x=model_input_iden, forward_type = 'identity',feature_grad=False)
        #out_iden =torch.sigmoid(predict_y_iden)
        #predict_y_iden, _ =torch.max(out_iden,dim=1)
        #print("predict_y_iden:",predict_y_iden.shape)
        #print("batch_y_iden:",batch_y_iden.shape)
        loss2=loss_funct_identity(predict_y_iden,batch_y_iden.long())
        
            
        optimizer2.zero_grad() # zero gradient
        loss2.backward()
        optimizer2.step()    # update gradient 
        
        
        data_source = train_iterator.next()
        batch_x, batch_y = data_source
        batch_x = batch_x[:,:,:-1] # [batch_size,128,19-1]
        batch_x = use_cuda(batch_x.float()) 
        batch_y = use_cuda(batch_y.float())
        model_input = batch_x.view(BATCH_SIZE,1,FREQUENCY*time_window_size,-1)
        #print(f'test 0.5')
        predict_y = model(model_input)
        out = torch.sigmoid(predict_y)
        predict_y, _ = torch.max(out,dim=1)
        loss = loss_function(predict_y,batch_y)
        
        for param in model.classifier2.parameters():
            param.requires_grad = False
        predict_y_iden = model(x=model_input_iden, forward_type = 'identity')
        #out_iden =torch.sigmoid(predict_y_iden)
        #predict_y_iden, _ =torch.max(out_iden,dim=1)
        loss2=loss_funct_identity(predict_y_iden,batch_y_iden.long())
        
        #optimizer.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params.lr)
        optimizer.zero_grad() # zero gradient
        task_loss=loss-params.lam*loss2
        task_loss.backward()
        optimizer.step()    # update gradient 
       
       
        predict_y_iden = model(x=model_input_iden, forward_type = 'identity',feature_grad=False)
        #out_iden =torch.sigmoid(predict_y_iden)
        #predict_y_iden, _ =torch.max(out_iden,dim=1)
        pred = torch.max(predict_y_iden, 1)[1]
        num_correct = (pred == batch_y_iden).sum()
        eval_acc += num_correct.item()
       
       
        for param in model.classifier2.parameters():
            param.requires_grad = True
        
        if i == 0:
            pass
      
        functional.reset_net(model)

        train_loss += loss.item()
        if loss.item() > -100 and loss.item() < 1e8:
            mean_loss_list.append(loss.item())
            if len(mean_loss_list)>1000:
                mean_loss_list.clear()
        mean_loss = np.mean(mean_loss_list)
        #pbar.set_postfix({'mean_loss':mean_loss})
        #pbar.update(1)

        label_list.extend(batch_y.reshape(-1).tolist())
        predict_list.extend(predict_y.data.cpu().tolist())
        use_print = 1
    print('Identity train acc: {:.6f}'.format(eval_acc / (len_train_iter*params.batch_size) * 100.))
    #print("predict_list:",predict_list)
    predict_label = [1 if p>params.threshold else 0 for p in predict_list]
    #check_result = [(label_list[i],predict_label[i]) for i in range(100)]
    accuracy = round(accuracy_score(label_list,predict_label),4)
    precision = round(precision_score(label_list,predict_label),4)
    recall = round(recall_score(label_list,predict_label),4)
    F1_score = round(f1_score(label_list,predict_label),4)
    train_auc = round(roc_auc_score(label_list,predict_list),4)
    specificity = round(specificity_score(label_list,predict_label),4)
    log.logger.info(f'epoch {epoch_i} end, train auc is {train_auc} ,train_loss is {round(train_loss,4)} ')
    log.logger.info(f'accuracy is {accuracy}, --- train recall is {recall} --- , f1 score is {F1_score} ')

    return accuracy,precision,recall,specificity,train_auc, train_loss, model 

def evaluate(model,eval_iterator,params,epoch_i,run_mode='eval'):
    log.logger.info(f'-- epoch is {epoch_i}, run_mode is {run_mode}, begin -- ')
    model.eval()
    eval_loss = 0.0
    label_list = []
    predict_list = []
    for i,(batch_x,batch_y) in enumerate(eval_iterator):
        #print("i:",i)
        #print("batch_x:",batch_x)
        #print("batch_y:",batch_y)
        batch_x = batch_x[:,:,:-1] # [batch_size,128,19-1]
        batch_x = use_cuda(batch_x.float())
        batch_y = use_cuda(batch_y.float())
        model_input = batch_x.view(BATCH_SIZE,1,FREQUENCY*time_window_size,-1)
       # model_input=model_input.view(BATCH_SIZE, 1,time_window_size,FREQUENCY,-1)# [batchsize, 1, 2,50, 18]
       # model_input=model_input.permute(2,0,4,3,1)
        predict_y = model(model_input)
        #predict_y = model(model_input).mean(0)
        out=torch.sigmoid(predict_y)
        predict_y,_=torch.max(out,dim=1)
        #print("predict_y",predict_y)
        loss = loss_function(predict_y,batch_y)
        eval_loss += loss.item()
        label_list.extend(batch_y.reshape(-1).tolist())
        predict_list.extend(predict_y.data.cpu().tolist())

    predict_label = [1 if p>params.threshold else 0 for p in predict_list]
    #check_result = [(label_list[i],predict_label[i]) for i in range(100)]
    accuracy = round(accuracy_score(label_list,predict_label),4)
    precision = round(precision_score(label_list,predict_label),4)
    recall = round(recall_score(label_list,predict_label),4)
    F1_score = round(f1_score(label_list,predict_label),4)
    eval_auc = round(roc_auc_score(label_list,predict_list),4)
    specificity = round(specificity_score(label_list,predict_label),4)
    log.logger.info(f'run mode is {run_mode}, epoch i is {epoch_i}, {run_mode} auc is {eval_auc} , {run_mode} loss is {eval_loss} ')
    log.logger.info(f'accuracy is {accuracy}, --- {run_mode} recall is {recall} ---, f1 score is {F1_score}, show confusion matrix below ')
    #get_confusion_matrix(label_list, predict_label)
    print('log_path:',log_path)
    return accuracy,precision,recall,specificity,eval_auc, eval_loss



def run_train(model, optimizer,use_eval=1):
    start_time = time.time()
    log.logger.info(f'run begin, start time is {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) } ')
    start_time_1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    train_auc_list = []
    eval_auc_list = []
    # sensitivity, specificity
    train_result_df = pd.DataFrame(columns=['accuracy','precision','recall','specificity','AUC','loss'])
    eval_result_df = pd.DataFrame(columns=['accuracy','precision','recall','specificity','AUC','loss'])

    best_model = model 
    best_score = 0
    best_epoch = 0

    log.logger.info(f'total epochs is {EPOCHS} ')

    CNN_Data.load_train_data()
    train_iterator = CNN_Data.train_loader
    eval_iterator = CNN_Data.eval_loader
    #for x in eval_iterator:
    #    print("x:",x)
    for i in range(EPOCHS):
        # model,optimizer,train_iterator,params,epoch_i
        train_iterator = iter(CNN_Data.train_loader)
        train_len=CNN_Data.train_len
        #train_identity_iterator=iter(CNN_Data_identity.train_loader)
        train_identity_len=CNN_Data_identity.train_len
        len_dataloader =int(train_identity_len/params.batch_size)
        accuracy,precision,recall,specificity,train_auc, train_loss, tmp_model = train_once_DANN(model, optimizer, optimizer2, train_iterator,CNN_Data_identity.train_loader,len_dataloader,params,i)
        train_auc_list.append(train_auc)
        train_result_df.loc[i]=[accuracy,precision,recall,specificity,train_auc,train_loss]
        if use_eval:
            accuracy,precision,recall,specificity,eval_auc, eval_loss = evaluate(model,eval_iterator,params,i)
            eval_auc_list.append(eval_auc)
            eval_result_df.loc[i]=[accuracy,precision,recall,specificity,eval_auc,eval_loss]

            #all_score = np.sum([accuracy,precision,recall*2,specificity,eval_auc])
            all_score = recall * 1.5 + specificity + eval_auc
            if all_score > best_score:
                model_path = os.path.join(model_root_path,f'pretrain_bestmodel_{start_time_tag}.pkl')
                torch.save(model, model_path)
                #print(f'improved, save epoch {i} model to {model_path} ')
                best_score = all_score
                best_epoch = i 
                #best_model = copy.deepcopy(tmp_model)  
                best_model.best_epoch = i        
    best_model=torch.load(f'/data/zzp/result/model_save/mit/pretrain_bestmodel_{start_time_tag}.pkl')        
    torch.cuda.empty_cache()
    end_time = time.time()
    if use_eval == 0:
        return best_model
    #log.logger.info(f'run end, start time is {start_time_1 } ')
    #log.logger.info(f'run end, end time is {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) } ')
    print(f'all run time is {round((end_time-start_time)/60,2)} min ')
    print(f'train_auc list is \n {train_auc_list} ')
    print(f'eval_auc list is \n {eval_auc_list} ')
    print(f'train result df is ')
    print(train_result_df.tail(10))
    print(f'eval result df is ')
    print(eval_result_df.tail(20))
    log.logger.info(f'----train-eval best score is {best_score}, best_epoch is {best_epoch} ----')
    #print(f'test result dic is :\n {test_result_dic} ')
    use_scale = params.use_scale
    best_model_save_path = os.path.join(model_root_path,f'resnet_scale{use_scale}_epoch{best_epoch}of{EPOCHS}_{start_time_tag}.pkl')
    torch.save(best_model, best_model_save_path)
    log.logger.info(f'-------------- best_pretrain_model_path save to : \n {best_model_save_path} ')
    return best_model_save_path

from utils import get_name_pinyin
def run_test_once(best_model):
    optimizer = torch.optim.Adam(best_model.parameters(),lr=params.lr)
    test_num = CNN_Data.test_num 
    test_result_df = pd.DataFrame(columns=['patient','Sensitivity','Specificity','Accuracy','AUC']) 
    for i in range(test_num):
        test_loader = CNN_Data.load_test_data(patient_id=i)
        patient_name = CNN_Data.test_name_list[i]
        name_py = patient_name#get_name_pinyin(patient_name)
        print(f'i is {i}, name is {patient_name}, py is {name_py} ')
        
        test_accuracy, test_precision, test_sensitivity, test_specificity, test_auc, _ = evaluate(best_model,
                                                                                                  test_loader, params,
                                                                                                  run_mode='test',
                                                                                                  epoch_i=i)
        test_result_df.loc[i] = [name_py, test_sensitivity, test_specificity, test_accuracy, test_auc]

    mean_result_list = ['mean']
    for col in test_result_df.columns[1:]:
        # print(f'col is {col}')
        m = test_result_df[col].mean()
        mean_result_list.append(m)
    test_result_df.loc[test_num] = mean_result_list

    #print(f'show test result, use_scale is {params.use_scale}, use pretrain is {params.use_pretrain} ')
    #print(test_result_df)

    #print(f'run test end')
    return test_result_df


def run_kfold_test(best_pretrain_model_path):
    log.logger.info(f'run kfold test begin, best model path is {best_pretrain_model_path} ')
    best_model = torch.load(best_pretrain_model_path)
    #optimizer = torch.optim.Adam(best_model.parameters(),lr=0.001)
    optimizer = torch.optim.Adam(best_model.parameters(),lr=params.lr)
    test_num = CNN_Data.test_num 
    kfold_test_result_df = pd.DataFrame(columns=['patient','Sensitivity','Specificity','Accuracy','AUC']) 
    fold_n = params.fold_n
    log.logger.info(f'run kfold test begin, fold n is {fold_n} ')
    for i in range(fold_n):
        base_model = copy.deepcopy(best_model)
        result_df = run_test_once(base_model)
        kfold_test_result_df = pd.concat([kfold_test_result_df, result_df])
        print(kfold_test_result_df.shape)

    #path = '/mnt/data1/user/kenan/result/kfold_result_df.pkl'
    #kfold_test_result_df.to_pickle(path)
    #print(kfold_test_result_df)
    final_result_df = merge_result(kfold_test_result_df)
    print(f'show final result df ')
    print(final_result_df)

def merge_result(kfold_df):
    #path = '/mnt/data1/user/kenan/result/kfold_result_df.pkl'
    #kfold_df = pd.read_pickle(path)
    name_list = list(kfold_df['patient'].unique())
    #print(name_list)
    #print(kfold_df)
    #for name in name_list:
    df = kfold_df.groupby(['patient']).mean()

    new_df = pd.DataFrame(columns=['patient','Sensitivity','Specificity','Accuracy','AUC']) 
    for i,index in enumerate(list(df.index)):
        if index == 'mean':
            continue
        new_df.loc[i] = [index] + list(df.loc[index].values)

    #print(new_df)
    #print(df)
    test_num = len(list(new_df.index))
    mean_result_list = ['mean']
    for col in new_df.columns[1:]:
        #print(f'col is {col}')
        m = new_df[col].mean()
        mean_result_list.append(m)
    new_df.loc[test_num] = mean_result_list
    #print(new_df)
    return new_df


if __name__ == '__main__':
    cuda_id = params.cuda
    
    with torch.cuda.device(cuda_id):
        config = {}
        config['MODEL'] = {}
        config['MODEL']['c_in'] = 1
        config['MODEL']['c_hidden'] = 64
        config['MODEL']['h_hidden'] = 50
        config['MODEL']['w_hidden'] = 9
        config['MODEL']['c_s1'] = 256
        config['MODEL']['c_s2'] = 384
        config['MODEL']['c_s3'] = 384
        config['MODEL']['c_s4'] = 256
        config['MODEL']['num_classes'] = 2
        config['MODEL']['kernel_size_x'] = 5
        config['MODEL']['pooling_x'] = False
        config['MODEL']['stride_x'] = 1
        config['MODEL']['padding_x'] = 2
        config['MODEL']['kernel_size_s'] = 5
        config['MODEL']['threshold'] = 0#30
        config['MODEL']['time_step'] = 2
        config['MODEL']['vth'] = 2.0
        config['MODEL']['dropout'] = 0.2
        config['MODEL']['leaky'] = 0.95
        config['MODEL']['identiClasses']= params.idenNum
        config['OPTIM'] = {}
        config['OPTIM']['solver'] = 'broy'
        config['MODEL']['input_type'] = 'constant'
        model = SNNIDEConvMultiLayerNet(config)
        model = model.cuda()
        module_list = []
        for name, p in model.named_parameters():
          if not ('classifier2' in name):
            module_list.append(p)
        #optimizer = torch.optim.SGD(module_list, lr=params.lr, momentum=0.9, weight_decay=5e-4)
        #optimizer2 = torch.optim.SGD(model.classifier2.parameters(), lr=params.lr, momentum=0.9, weight_decay=5e-4)
       
        if params.optimizer=='Adam':
          optimizer = torch.optim.Adam(module_list,lr=params.lr)#without weightdecay
          optimizer2 = torch.optim.Adam(model.classifier2.parameters(),lr=params.lr)
        
        elif params.optimizer=='AdamWD':
          optimizer = torch.optim.Adam(module_list,lr=params.lr,weight_decay=5e-4)#with weightdecay
          optimizer2 = torch.optim.Adam(model.classifier2.parameters(),lr=params.lr,weight_decay=5e-4)
        elif  params.optimizer=='SGDMWD':  
          optimizer = torch.optim.SGD(module_list, lr=params.lr, momentum=0.9, weight_decay=5e-4)
          optimizer2 = torch.optim.SGD(model.classifier2.parameters(), lr=params.lr, momentum=0.9, weight_decay=5e-4)
        
        #best_pretrain_model_path = '/data/zzp/result/model_save/mit/resnet_scale0_epoch12of50_12122255.pkl'

        
        best_pretrain_model_path = run_train(model, optimizer=optimizer, use_eval=1)###
        print('log_path:',log_path)
        #best_model = torch.load(best_pretrain_model_path)
        run_kfold_test(best_pretrain_model_path)
        log.logger.info(f'-------------- best_pretrain_model_path is : \n {best_pretrain_model_path} ')
        #run_test_once(best_model)
        #run_kfold_test(best_model)
    #merge_result()
    end_time_tag = time.strftime("%m%d%H%M")
    log.logger.info(f'start time tag is {start_time_tag}, end_time_tag is {end_time_tag} ')
    log.logger.info(f'log path is {log_path} ')




