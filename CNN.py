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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, Concatenate, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from keras import backend as K

cli=None
translator=None


#To use F1 score as metric
#https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

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
        clean.append([datetime.strptime(each[:12], "%b %d %H:%M"), [x.lower() for x in  re.sub(r'[0-9]+',
            '', each[each.find(':',19)+1:]).translate(translator).split() ]])
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

    '''try :
        text = re.sub(r'[0-9]+', '', text)
    except Exception as e:
        print("Operation error at re.sub : %s"%e)
        os.remove(local_path+file_name)
        return ''
    '''
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

def generator(inputs, labels):
    #to fit with different input dimension
    #https://github.com/keras-team/keras/issues/1920#issuecomment-410982673
    i = 0
    while True:
        inputs_batch = np.expand_dims([inputs[i%len(inputs)]], axis=2)
        labels_batch = np.array([labels[i%len(inputs)]])
        yield inputs_batch, labels_batch
        i+=1

def cnn_learning(model,X,y,max_len):
    train_num=int(len(X)*0.8)
    test_num=len(X)-train_num
    #X=pad_sequences(X, maxlen=max_len)
    #X_test=[np.array(x, dtype='float').reshape(len(x),1) for x in X[train_num:]]
    X=np.array(X)
    X_test=X[train_num:]
    X_train=X[:train_num]
    y_test=np.array(y[train_num:])
    #X_train=[np.array(x, dtype='float').reshape(len(x),1) for x in X[:train_num]]
    y_train=np.array(y[:train_num])
    '''print('X_train의 크기(shape) :',X_train.shape)
    print('X_test의 크기(shape) :',X_test.shape)
    print('y_train의 크기(shape) :',y_train.shape)
    print('y_test의 크기(shape) :',y_test.shape)'''
    X_valid=X_train[int(0.8*train_num):]
    y_valid=y_train[int(0.8*train_num):]
    X_train=X_train[:int(0.8*train_num)]
    y_train=y_train[:int(0.8*train_num)]
    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 0, patience = 3)
    mc = ModelCheckpoint('best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 0, save_best_only = True)
    #history = model.fit(X_train, y_train, epochs = 40, validation_split = 0.2, callbacks=[es, mc], verbose=1)
    model.fit_generator(
    generator(X_train, y_train),
    validation_data=generator(X_valid, y_valid),
    steps_per_epoch=len(X_train),
    validation_steps=len(X_valid),
    epochs=10, verbose=1, callbacks=[es,mc])

    return X_test, y_test


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--fault', type=str, help='Fault history file path')
    parser.add_argument('--model', type=str, help='CNN model file path')
    parser.add_argument('--emb', type=str, help='Word Embedding file path', default='embedding_with_log')
    args=parser.parse_args()
    local_path='/home/dpnm/tmp/'
    remote_path='/mnt/hdd/log/'#225-2c-4/04-23/'
    model_file=False
    #file_name='snort.log'
    vnf_list=[
        {'vnf_num' : 1, 'vnf_id' : 'b21b38dc-eee0-4f27-9863-1b600705a126', 'vnf_name' : '226-4c-1', 'ip': '10.10.10.226'},
        {'vnf_num' : 2, 'vnf_id' : '3f97a1f5-f766-4b9c-89dd-ef7f8ef9adb7', 'vnf_name' : '225-4c-1', 'ip' : '10.10.10.148'},
        {'vnf_num' : 3, 'vnf_id' : '65d6314d-c3cc-46d0-9856-67fc95b953f7', 'vnf_name' : '225-2c-1', 'ip' : '10.10.10.126'},
        {'vnf_num' : 4, 'vnf_id' : '75389a36-510b-41ac-9a19-380b9716c30c', 'vnf_name' : '225-2c-2', 'ip' : '10.10.10.14'},
        {'vnf_num' : 5, 'vnf_id' : 'cbe5d6e7-a490-4326-aea4-8035dc8b3d46', 'vnf_name' : '225-2c-3', 'ip' : '10.10.10.124'},
        {'vnf_num' : 6, 'vnf_id' : 'dc5422d7-9e9d-4dc4-a381-5e2bcc986667', 'vnf_name' : '225-2c-4', 'ip' : '10.10.10.26'}]
    translator = str.maketrans('', '', string.punctuation)
    cli=paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    if not args.fault:
        print("-------------Get Fault history-----------------")
        fault_history={}
        with open ('../server_info.yaml') as f:
            server_info=yaml.load(f)['BootingHistory']
        cli.connect(server_info['ip'], port=22, username=server_info['id'], password=server_info['pwd'])
        try:
            with SCPClient(cli.get_transport()) as scp:
                scp.get('/home/ubuntu/booting_history.yaml',local_path)
        except SCPException:
            raise SCPException.message
        with open(local_path+'booting_history.yaml')as f:
            fault_history=yaml.load(f)
        for vnf in vnf_list:
            fault_ori=fault_history[vnf['vnf_num']]
            fault_new=[]
            for fault_time in fault_ori:
                fault_new.append([datetime.strptime(fault_time, "%b-%d-%H:%M").strftime("%m-%d %H:%M"), 'reboot'])
            fault_new.extend([[x, 'ppt over'] for x in fault_tagging(vnf['vnf_num'])])
            fault_history[vnf['vnf_num']]=fault_new
        with open('fault_history.yaml','w') as f:
            yaml.dump(fault_history,f)
        #print(fault_history)
        print("-------------Get Fault history END-------------")
    else:
        with open(args.fault)as f:
            fault_history=yaml.load(f)
    wv= Word2Vec.load(args.emb)
    vocab_size=len(wv.wv)
    embed_size=100
    max_len=20000
    with open ('../server_info.yaml') as f:
        server_info=yaml.load(f)['log']
    cli.connect(server_info['ip'], port=22, username=server_info['id'], password=server_info['pwd'])
    if not args.model:
        model_input=Input(shape=(None,1))
        #model=Sequential()
        #model.add(Embedding(vocab_size, 32))
        #model.add(Dropout(0.2))
        submodels=[]
        for kw in (3,4,5):
            conv=Conv1D(100, kw*embed_size, padding='valid', activation='relu', kernel_regularizer=l2(3),strides=embed_size)(model_input)
            conv=GlobalMaxPooling1D()(conv)
            #conv=Flatten()(conv)
            submodels.append(conv)
        z=Concatenate()(submodels)
        z=Dropout(0.5)(z)
        model_output=Dense(1,activation='softmax')(z)
        model=Model(model_input, model_output)
        model.summary()
        model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['acc',f1_m,precision_m, recall_m])
    else:
        model = load_model(args.model)
    #tf.debugging.set_log_device_placement(True)
    #devices = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(devices[0], True)
    win_size=20
    gap=3
    sliding=1

    log_corpus=[]
    test_X=[]
    test_y=[]
    X_len=0
    fault_len=0
    today=datetime.today().strftime("%m-%d")
    for vnf in vnf_list:
        vnf_=vnf['vnf_name']
        print('Reading %d th VNF, %s'%(vnf['vnf_num'], vnf_))
        log_dir_list=date_range("04-21",today)
        for j in trange(len(log_dir_list)):
            log_dir=log_dir_list[j]
            log_file_list=get_file_list(remote_path+vnf_+'/'+log_dir)
            path_=remote_path+vnf_+'/'+log_dir+'/'
            for file_name in log_file_list:
                if file_name in ['sudo.log', 'CRON.log', 'stress-ng.log', 'apache-access.log']:
                    continue
                log = download_log(path_,file_name,local_path)
                log_token=pre_process(log)
                log_corpus.extend(log_token)
            log_corpus=sorted(log_corpus, key= lambda x : x[0])
            date=datetime.strptime(log_dir,'%m-%d')
            X=[]
            y=[]
            fault_num=0
            local_max_len=0
            while(date<datetime.strptime(log_dir,'%m-%d')+timedelta(days=1)):
                input_log=[]
                for log in log_corpus[:]:
                    if date>log[0]:
                        log_corpus.remove(log)
                    if log[0] > date+timedelta(minutes=win_size):
                        break
                    for word in log[1]:
                        try:
                            input_log.extend(wv.wv.get_vector(word))
                        except:
                            input_log.extend(np.zeros(embed_size))
                assert len(input_log)%embed_size==0
                if not len(input_log)==0:
                    if len(input_log)<5*embed_size:
                        #input_log=pad_sequences(input_log, maxlen=5*embed_size).tolist()
                        input_log.extend(np.zeros(5*embed_size-len(input_log)))
                    local_max_len = local_max_len if local_max_len > len(input_log) else len(input_log)
                    #if(len(input_log)>max_len):
                    #    input_log=input_log[:max_len+1]
                    X.append(input_log)
                    if (date+timedelta(minutes=gap)).strftime( '%m-%d %H:%M') in [x[0] for x in fault_history[vnf['vnf_num']]]:
                        y.append(1)
                        y.append(1)
                        X.append(input_log) #For over Sampling
                        date+=timedelta(minutes=gap) ## Slide to after of Fault
                        fault_num+=2
                        continue
                    else:
                        y.append(0)
                date+=timedelta(minutes=sliding)
            print('\nlocal max : %d'%local_max_len)
            with open('data_gap%d_win%d.bin'%(gap, win_size), 'a+b') as f:
                pkl.dump([X,y], f)
            print('\n%d number of data created and %d number of them are fault'%(len(X), fault_num))
            X_len+=len(X)
            fault_len+=fault_num
            X,y=cnn_learning(model, X,y,max_len)
            test_X.extend(X)
            test_y.extend(y)
            log_corpus=[]
        loss, accuracy, f1_score, precision, recall = model.evaluate(test_X, test_y)
        print("Read total %d number of data and %d of them are fault"%(X_len, fault_len))
        print("F1 : %.4f, Acc : %.4f, Prec : %.4f, Rec : %.4f"%(f1_score, accuracy, precision, recall))
        model.save("cnn_gap%d_win%d"%(gap, win_size))
