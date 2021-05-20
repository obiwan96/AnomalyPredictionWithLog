import nltk
from gensim.models.word2vec import Word2Vec
import string
import pickle as pkl
from string import digits
from datetime import datetime, timedelta
import scp
import paramiko
from scp import SCPClient, SCPException
import os
import re
import argparse
from tqdm import trange
from gensim.models import KeyedVectors
import pandas as pd
from influxdb import DataFrameClient
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import random
import Model

cli=None
translator=None
embed_size=100
vnf_list=[
    {'vnf_num' : 1, 'vnf_id' : 'b21b38dc-eee0-4f27-9863-1b600705a126', 'vnf_name' : '226-4c-1', 'ip': '10.10.10.226'},
    {'vnf_num' : 2, 'vnf_id' : '3f97a1f5-f766-4b9c-89dd-ef7f8ef9adb7', 'vnf_name' : '225-4c-1', 'ip' : '10.10.10.148'},
    {'vnf_num' : 3, 'vnf_id' : '65d6314d-c3cc-46d0-9856-67fc95b953f7', 'vnf_name' : '225-2c-1', 'ip' : '10.10.10.126'},
    {'vnf_num' : 4, 'vnf_id' : '75389a36-510b-41ac-9a19-380b9716c30c', 'vnf_name' : '225-2c-2', 'ip' : '10.10.10.14'},
    {'vnf_num' : 5, 'vnf_id' : 'cbe5d6e7-a490-4326-aea4-8035dc8b3d46', 'vnf_name' : '225-2c-3', 'ip' : '10.10.10.124'},
    {'vnf_num' : 6, 'vnf_id' : 'dc5422d7-9e9d-4dc4-a381-5e2bcc986667', 'vnf_name' : '225-2c-4', 'ip' : '10.10.10.26'},
    #server
    {'vnf_num' : 225, 'vnf_id' : 'server', 'vnf_name' : 'dpnm-82-225', 'ip' : ''}]

def pre_process(text):
    #https://blog.naver.com/PostView.nhn?blogId=timtaeil&logNo=221361106051&redirect=Dlog&widgetTypeCall=true&directAccess=false
    #Delete former words of ':' Because it contain dates and host name.
    #Example : Mar 23 06:37:33 225-2c-1 10freedos: debug: /dev/vda15 is a FAT32 partition
    #Remove number, symbol, stop world
    #Tokenizing
    #clean = [[x.lower() for x in each[each.find(':',19)+1:].translate(translator).split() \
    #       if x.lower() not in stop_words] for each in text.split('\n')]
    if text=='':
        return []
    global translator
#    text = re.sub(r'[0-9]+', '', text)
    clean =[]
    for each in text.split('\n'):
        if not each:
            continue
        #Todo : How to use number information??
        clean.append(tuple([datetime.strptime(each[:12], "%b %d %H:%M"), tuple([x.lower() for x in  re.sub(r'[0-9]+',
            '', each[each.find(':',19)+1:]).translate(translator).split()] )]))
    return clean
def pre_process_error_only(text):
    if text=='':
        return []
    global translator
    clean =[]
    for each in text.split('\n'):
        if not 'error' in each:
            continue
        #Todo : How to use number information??
        clean.append(tuple([datetime.strptime(each[:12], "%b %d %H:%M"), tuple([x.lower() for x in  re.sub(r'[0-9]+',
            '', each[each.find(':',19)+1:]).translate(translator).split() ])]))
    return clean

def download_log(remote_path, file_name, local_path):
    global cli
    try:
        with SCPClient(cli.get_transport()) as scp:
            scp.get(remote_path+file_name, local_path)
    except SCPException as e:
        print("Operation error : %s"%e)
    try:
        with open(local_path+file_name) as f:
            text = f.read()
    except:
        try:
            with open(local_path+file_name, encoding='ISO-8859-1') as f:
                text = f.read()
        except Exception as e:
            print("Opreation error at reading file %s : %s"%(file_name, e))
    os.remove(local_path+file_name)
    return text

def get_file_list(path):
    global cli
    stdin, stdout, stderr = cli.exec_command('ls '+path)
    rslt=stdout.read()
    file_list=rslt.split()
    del stdin, stdout, stderr
    file_list = [file_name.decode('utf-8') for file_name in file_list]
    return file_list

def date_range(start, end):
    start=datetime.strptime(start, "%m-%d")
    end = datetime.strptime(end, "%m-%d")
    dates = [(start + timedelta(days=i)).strftime("%m-%d") for i in range((end-start).days+1)]
    return dates

def fault_tagging(vnf_num):
    #Not Use
    global cli
    with open ('../server_info.yaml') as f:
        server_info=yaml.load(f)['InDB']
    user, password, host = server_info['id'], server_info['pwd'], server_info['ip']
    client=DataFrameClient(host, 8086,user, password, 'pptmon')
    ppt = client.query('select * from "%d"'%vnf_num)
    ppt=list(ppt.values())[0].tz_convert('Asia/Seoul')
    ppt.index=ppt.index.map(lambda x : x.replace(microsecond=0, second=0))
    ppt.reset_index(inplace = True)
    ppt.rename(columns={'index' : 'time'}, inplace=True)
    fault= ppt[ppt['value']>10000][['time']].values.tolist()
    fault = [x[0].strftime("%m-%d %H:%M") for x in fault]
    return fault

def get_fault_history(vnf_num):
    print("-------------Get Fault history-----------------")
    fault_history={}
    with open ('../server_info.yaml') as f:
        server_info=yaml.load(f)['BootingHistory']
    cli.connect(server_info['ip'], port=22, username=server_info['id'], password=server_info['pwd'])
    try:
        with SCPClient(cli.get_transport()) as scp:
            scp.get('/home/ubuntu/fault_history.yaml',local_path)
    except SCPException:
        raise SCPException.message
    with open(local_path+'fault_history.yaml')as f:
        fault_history=yaml.load(f)
    return fault_history[vnf_num]['abnormal']+fault_history[vnf_num]['fault']

def generator(inputs, labels):
    #to fit with different input dimension
    #https://github.com/keras-team/keras/issues/1920#issuecomment-410982673
    i = 0
    while True:
        inputs_batch = np.expand_dims([inputs[i%len(inputs)]], axis=2)
        labels_batch = np.array([labels[i%len(inputs)]])
        yield inputs_batch, labels_batch
        i+=1

def model_learning_with_validation(model,X,y, verbose=0):
    #shuffle the data
    assert len(X)==len(y)
    s=np.arange(len(X))
    X=np.array(X)
    y=np.array(y)
    np.random.shuffle(s)
    X=X[s]
    y=y[s]
    train_num=int(len(X)*0.8)
    test_num=len(X)-train_num
    X_test=X[train_num:]
    X_train=X[:train_num]
    y_test=np.array(y[train_num:])
    y_train=np.array(y[:train_num])
    X_valid=X_train[int(0.8*train_num):]
    y_valid=y_train[int(0.8*train_num):]
    X_train=X_train[:int(0.8*train_num)]
    y_train=y_train[:int(0.8*train_num)]
    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 0, patience = 3)
    mc = ModelCheckpoint('best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 0, save_best_only = True)
    model.fit_generator(
        generator(X_train, y_train),
        validation_data=generator(X_valid, y_valid),
        steps_per_epoch=len(X_train),
        validation_steps=len(X_valid),
        epochs=1000, verbose=verbose, callbacks=[mc,es])
    return model, X_test, y_test

def model_learning(model,X,y, verbose=0):
    assert len(X)==len(y)
    X=np.array(X)
    y=np.array(y)
    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 0, patience = 3)
    mc = ModelCheckpoint('best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 0, save_best_only = True)
    model.fit_generator(generator(X, y), steps_per_epoch=len(X), epochs=1000, verbose=verbose, callbacks=[mc,es])
    return model

def make_data(vnf, fault_, win_size, gap, date_list, sliding=1, use_nolog=False, over_sampling=1, under_sampling=0):
    same_limit=5
    X=[]
    y=[]
    fault_len=0
    log_dir_list=get_file_list(remote_path+vnf['vnf_name']+'/')
    for j in trange(len(date_list)):
        log_dir = date_list[j]
        if not log_dir in log_dir_list:
            continue
        path_=remote_path+vnf['vnf_name']+'/'+log_dir+'/'
        log_file_list=get_file_list(path_)
        log_corpus=[]
        for file_name in log_file_list:
            if vnf['vnf_id']=='server':
                if file_name in ['sudo.log', 'CRON.log', 'nova-compute.log', 'neutron-openvswitch-agent.log']:
                    continue
            elif file_name in ['sudo.log', 'CRON.log', 'stress-ng.log']:
                continue
            log = download_log(path_,file_name,local_path)
            if file_name=='kernel.log':
                log_token = pre_process_error_only(log)
            else:
                log_token=pre_process(log)
            log_corpus.extend(log_token)

        #Delete Same log. This could be dangerous.
        log_corpus=sorted(list(set(log_corpus)), key= lambda x : x[0])
        date=datetime.strptime(log_dir,'%m-%d')
        fault_num=0
        while(date<datetime.strptime(log_dir,'%m-%d')+timedelta(days=1)):
            input_log=[]
            sentence_pool=[]
            for log in log_corpus[:]:
                if date>log[0]:
                    log_corpus.remove(log)
                if log[0] > date+timedelta(minutes=win_size):
                    break
                ######If first words are same, it seems similar log. pass it####
                already_in_sentence_pool=False
                for sentence in sentence_pool:
                    if sentence==list(log[1])[:same_limit]:
                        already_in_sentence_pool=True
                        break
                if already_in_sentence_pool:
                    continue
                sentence_pool.append(list(log[1])[:same_limit])
                ######Checking similar log end####
                for word in log[1]:
                    try:
                        input_log.extend(wv.wv.get_vector(word))
                    except:
                        pass #Do not add OOV
                        #input_log.extend(np.zeros(embed_size))
            if use_nolog and len(input_log) ==0:
                input_log=np.zeros(5*embed_size)
            assert len(input_log)%embed_size==0
            fault=False
            for i in range(sliding):
                if (date+timedelta(minutes=gap+win_size+i)).strftime('%b-%d-%H:%M') in fault_:
                    fault=True
                    break
            if not len(input_log)==0:
                if len(input_log)<5*embed_size:
                    input_log.extend(np.zeros(5*embed_size-len(input_log)))
                if fault:
                    y.extend([1]*over_sampling)
                    X.extend([input_log for _ in range(over_sampling)]) #For over Sampling
                    date+=timedelta(minutes=gap+win_size) ## Slide to after of Fault
                    fault_num+=over_sampling
                    continue
                else:
                    if random.choice([True]+[False for i in range(under_sampling)]):
                        X.append(input_log)
                        y.append(0)
            date+=timedelta(minutes=sliding)
        fault_len+=fault_num
    print("Read total %d number of data and %d of them are fault"%(len(X), fault_len))
    return X,y

def CNN_learning(model):
    test_all_X=[]
    test_all_y=[]
    today=datetime.today().strftime("%m-%d")
    date_list=date_range('05-11',today)
    gap=3
    win_size=20
    for vnf in vnf_list:
        vnf_=vnf['vnf_name']
        fault_=get_fault_history(vnf['vnf_num'])
        with open ('../server_info.yaml') as f:
            server_info=yaml.load(f)['log']
        cli.connect(server_info['ip'], port=22, username=server_info['id'], password=server_info['pwd'])
        print('Reading %d th VNF, %s'%(vnf['vnf_num'], vnf_))
        X,y=make_data(vnf,fault_,win_size,gap,date_list, over_sampling=10,under_sampling=400)
        model, test_X,test_y=model_learning_with_validation(model, X,y, verbose=0)
        test_all_X.extend(test_X)
        test_all_y.extend(test_y)
        loss, accuracy, f1_score, precision, recall = model.evaluate_generator(
                generator(test_X, test_y), steps= len(test_X))
        print("F1 : %.4f, Acc : %.4f, Prec : %.4f, Rec : %.4f"%(f1_score, accuracy, precision, recall))
        model.save("models/CNN_gap%d_win%d"%(gap, win_size))
        print("model saved")
    loss, accuracy, f1_score, precision, recall = model.evaluate_generator(
            generator(test_all_X, test_all_y), steps= len(test_all_X))
    print("-------Learning End. Final performance is here---------")
    print("F1 : %.4f, Acc : %.4f, Prec : %.4f, Rec : %.4f"%(f1_score, accuracy, precision, recall))

def CRNN_learning(model):
    gap=3
    win_size=5
    today=datetime.today().strftime("%m-%d")
    date_list=date_range('05-11',today)
    for vnf in vnf_list:
        vnf_=vnf['vnf_name']
        fault_=get_fault_history(vnf['vnf_num'])
        with open ('../server_info.yaml') as f:
            server_info=yaml.load(f)['log']
        cli.connect(server_info['ip'], port=22, username=server_info['id'], password=server_info['pwd'])
        print('Reading %d th VNF, %s'%(vnf['vnf_num'], vnf_))
        X,y= make_data(vnf, fault_, win_size,gap,date_list[:-1],sliding=5,use_nolog=True)
        model=model_learning(model, X,y, verbose=1)
        #Last day is the Test set
        test_X,test_y= make_data(vnf, fault_, win_size,gap,date_list[-1:],sliding=5,use_nolog=True)
        loss, accuracy, f1_score, precision, recall = model.evaluate_generator(
                generator(test_X, test_y), steps= len(test_X))
        print("F1 : %.4f, Acc : %.4f, Prec : %.4f, Rec : %.4f"%(f1_score, accuracy, precision, recall))
        model.save("models/CRNN_gap%d_win%d"%(gap, win_size))
        print("model saved")

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--type', type=str, help='ML type. CNN or CRNN')
    parser.add_argument('--model', type=str, help='model file path')
    parser.add_argument('--emb', type=str, help='Word Embedding file path', default='embedding_with_log')
    args=parser.parse_args()
    local_path='/home/dpnm/tmp/'
    remote_path='/mnt/hdd/log/'
    model_file=False
    translator = str.maketrans(string.punctuation, ' '*(len(string.punctuation)))
    cli=paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    wv= Word2Vec.load(args.emb)
    vocab_size=len(wv.wv)
    if not args.model:
        if args.type =='cnn':
            model = Model.CNN()
        elif args.type=='crnn':
            model = Model.CRNN()
    else:
        model = load_model(args.model, custom_objects={"f1_m":Model.f1_m, "precision_m":Model.precision_m, "recall_m":Model.recall_m})
    if args.type =='cnn':
        CNN_learning(model)
    elif args.type=='crnn':
        CRNN_learning(model)
