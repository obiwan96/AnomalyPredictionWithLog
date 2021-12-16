import argparse
import yaml
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from keras import backend as K
from datetime import datetime, timedelta
import Model
import os
#from now on, for data reading.
#need to delete if not use.
from tqdm import trange
import string
import pickle as pkl
from string import digits
from datetime import datetime, timedelta
from influxdb import DataFrameClient
import numpy as np
import pandas as pd
import yaml
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import re
import random

from Learning import generator
from DataReading_fromLocal import get_file_list, download_log, pre_process, pre_process_error_only, get_fault_history, date_range
local_path='/ddhome/dpnm/tmp/'
local_log_path='/home/dpnm/log/'
threshold=0.8

def make_data(vnf, fault_, model_parameter_list, date_list, max_prediction_object_length, sliding=1, use_emptylog=False):
    wv= Word2Vec.load('embedding_with_log')
    #wv= Word2Vec.load('embedding_with_log_with_underbar')
    embed_size=100
    #oov=np.array([1/embed_size**0.5]*embed_size)
    same_limit=5
    X=[]
    y=[]
    fault_len=0
    fault_end_date=None
    for j in range(len(date_list)):
        log_dir = date_list[j]
        if fault_end_date:
            if datetime.strptime(log_dir,'%m-%d') < datetime.strptime(fault_end_date[:fault_end_date.find('-',4)],'%b-%d'):
                continue
        path_=local_log_path+vnf['vnf_name']+'/'+log_dir+'/'
        log_file_list=get_file_list(path_)
        log_corpus=[]
        #download each day log
        for file_name in log_file_list:
            if file_name in ['sudo.log', 'CRON.log', 'stress-ng.log']:
                continue
            log = download_log(file_name,path_)
            if file_name in ['kernel.log', 'ovs-vswitchd.log']:
                log_token = pre_process_error_only(log)
            else:
                log_token=pre_process(log)
            log_corpus.extend(log_token)

        #Delete Same log. This could be dangerous.
        log_corpus=sorted(list(set(log_corpus)), key= lambda x : x[0])
        #date is prediction object time.
        #need to start from 12' max_prediction_object_length minutes m.
        if fault_end_date:
            date=datetime.strptime(fault_end_date,'%b-%d-%H:%M')
            fault_end_date=None
        else:
            date=datetime.strptime(log_dir,'%m-%d')
        date+=timedelta(minutes=max_prediction_object_length)
        while(date<datetime.strptime(log_dir,'%m-%d')+timedelta(days=1)):
            #delete if prediction object pass the corpus
            for log in log_corpus[:]:
                if date-timedelta(minutes=max_prediction_object_length)>log[0]:
                    log_corpus.remove(log)
                if log[0] > date:
                    break

            #make seperate data for each gap and window_size
            #but they should have same y, which is prediction object.
            x=[]
            for model_parameter in model_parameter_list:
                gap, win_size = model_parameter
                max_fil_size=win_size//2+3
                input_log=[]
                sentence_pool=[]
                for log in log_corpus[:]:
                    ######If first 5 words are same, it seems similar log. pass it####
                    if log[0] < date-timedelta(minutes=gap+win_size):
                        pass
                    if log[0] > date-timedelta(minutes=gap):
                        break
                    already_in_sentence_pool=False
                    for sentence in sentence_pool:
                        if sentence==list(log[1])[:same_limit]:
                            already_in_sentence_pool=True
                            break
                    if already_in_sentence_pool:
                        continue
                    sentence_pool.append(list(log[1])[:same_limit])
                    #now, add word embedding
                    for word in log[1]:
                        try:
                            input_log.extend(wv.wv.get_vector(word))
                        except:
                            #OOV
                            input_log.extend(np.random.rand(embed_size))
                if use_emptylog and len(input_log) ==0:
                    input_log=np.zeros((win_size)*embed_size)
                assert len(input_log)%embed_size==0
                abnormal=False
                fault=False
                for i in range(sliding):
                    if (date+timedelta(minutes=i)).strftime('%b-%d-%H:%M') in fault_['abnormal']:
                        abnormal=True
                        break
                    for fault_range in fault_['fault']:
                        if (date+timedelta(minutes=i)).strftime('%b-%d-%H:%M') == fault_range['start']:
                            fault=True
                            if 'end' in fault_range:
                                fault_end_date=fault_range['end']
                            break
                    if fault:
                        break
                if not len(input_log)==0:
                    if len(input_log)<max_fil_size*embed_size:
                        input_log.extend(np.zeros(max_fil_size*embed_size-len(input_log)))
                    x.append(input_log)
                else:
                    x.append([])
            if fault or abnormal:
                y.extend([1])
                X.extend(x)
                if fault:
                    if fault_end_date:
                        if datetime.strptime(log_dir,'%m-%d') < datetime.strptime(fault_end_date[:fault_end_date.find('-',4)],'%b-%d'):
                            #Go to next date.
                            break
                        else:
                            date=datetime.strptime(fault_end_date,'%b-%d-%H:%M')+timedelta(max_prediction_object_length)
                            fault_end_date=None
                            continue
                    else:
                        return X,y
                continue
            else:
                X.append(x)
                y.append(0)
            date+=timedelta(minutes=sliding)
    #print("Read total %d number of data and %d of them are fault"%(len(X), fault_len))
    return X,y

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', default='models/best_models/')
    args=parser.parse_args()
    best_models_name_list=os.listdir(args.model)
    best_models_list=[]
    model_parameter_list=[]
    max_prediction_object_length=0
    for model_name in best_models_name_list:
        if model_name.startswith('gap_'):
            tmp=model_name.split('_')
            model={}
            model['model']=load_model(args.model+model_name, custom_objects={"KL_dv_loss":Model.KL_dv_loss})
            gap=int(tmp[1])
            win_size=int(tmp[3])
            if gap+win_size>max_prediction_object_length:
                max_prediction_object_length=gap+win_size
            model['gap']=gap
            model['window_size']=win_size
            model_parameter_list.append((gap,win_size))
            best_models_list.append(model)
    print('load model file end')
    vnf_list=[
    {'vnf_num' : 1, 'vnf_id' : 'b21b38dc-eee0-4f27-9863-1b600705a126', 'vnf_name' : '226-4c-1', 'ip': '10.10.10.226'},
    {'vnf_num' : 2, 'vnf_id' : '3f97a1f5-f766-4b9c-89dd-ef7f8ef9adb7', 'vnf_name' : '225-4c-1', 'ip' : '10.10.10.148'},
    {'vnf_num' : 3, 'vnf_id' : '65d6314d-c3cc-46d0-9856-67fc95b953f7', 'vnf_name' : '225-2c-1', 'ip' : '10.10.10.126'},
    {'vnf_num' : 4, 'vnf_id' : '75389a36-510b-41ac-9a19-380b9716c30c', 'vnf_name' : '225-2c-2', 'ip' : '10.10.10.14'},
    {'vnf_num' : 5, 'vnf_id' : 'cbe5d6e7-a490-4326-aea4-8035dc8b3d46', 'vnf_name' : '225-2c-3', 'ip' : '10.10.10.124'},
    {'vnf_num' : 6, 'vnf_id' : 'dc5422d7-9e9d-4dc4-a381-5e2bcc986667', 'vnf_name' : '225-2c-4', 'ip' : '10.10.10.26'}
    ]
    all_y=[]
    all_predict=[]
    for vnf in vnf_list:
        vnf_name=vnf['vnf_name']
        date_list=date_range('05-11', '06-06', vnf_name)
        fault_list=get_fault_history(vnf['vnf_num'])
        X,y=make_data(vnf, fault_list,model_parameter_list, date_list, max_prediction_object_length)
        assert(length(X)==length(y))
        for i in range (lenth(X)):
            x=X[i]
            pred=0
            for model in best_model_list:
                if model['model'].predict_generator(generator([x], [y[i]]), steps=1)[0]>threshold:
                    pred=1
                    break
            all_predict.append(pred)
        all_y.extend(y)
    for i in range(len(all_y)):
        if all_predict[i]==all_y[i]:
            right+=1.0
        if all_predict[i]:
            if y_test[i]:
                tp+=1.0
            else:
                fp+=1.0
        elif all_y[i]:
            fn+=1.0
    accuracy = right/(len(y_pred)+K.epsilon())
    precision=tp/(tp+fp+ K.epsilon())
    recall=tp/(tp+fn+ K.epsilon())
    f1=2*recall*precision/(recall+precision+K.epsilon())
    if print_result:
        print('tp : %d'%(tp)+' fp : %d'%(fp)+ ' fn : %d'%(fn))
        print('acc : %4f'%(right/(len(y_pred)+ K.epsilon()))+' precision : %4f'%(precision)+
               ' recall : %4f'%(recall)+' f1 : %4f'%(f1))
        print('-----------------------------------------------')
