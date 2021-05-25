import nltk
import argparse
import yaml
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from keras import backend as K
from datetime import datetime, timedelta
import Model
from DataReading import *

vnf_list=[
    {'vnf_num' : 1, 'vnf_id' : 'b21b38dc-eee0-4f27-9863-1b600705a126', 'vnf_name' : '226-4c-1', 'ip': '10.10.10.226'},
    {'vnf_num' : 2, 'vnf_id' : '3f97a1f5-f766-4b9c-89dd-ef7f8ef9adb7', 'vnf_name' : '225-4c-1', 'ip' : '10.10.10.148'},
    {'vnf_num' : 3, 'vnf_id' : '65d6314d-c3cc-46d0-9856-67fc95b953f7', 'vnf_name' : '225-2c-1', 'ip' : '10.10.10.126'},
    {'vnf_num' : 4, 'vnf_id' : '75389a36-510b-41ac-9a19-380b9716c30c', 'vnf_name' : '225-2c-2', 'ip' : '10.10.10.14'},
    {'vnf_num' : 5, 'vnf_id' : 'cbe5d6e7-a490-4326-aea4-8035dc8b3d46', 'vnf_name' : '225-2c-3', 'ip' : '10.10.10.124'},
    {'vnf_num' : 6, 'vnf_id' : 'dc5422d7-9e9d-4dc4-a381-5e2bcc986667', 'vnf_name' : '225-2c-4', 'ip' : '10.10.10.26'}]
    #server
#    {'vnf_num' : 225, 'vnf_id' : 'server', 'vnf_name' : 'dpnm-82-225', 'ip' : ''},
#    {'vnf_num' : 226, 'vnf_id' : 'server', 'vnf_name' : 'dpnm-82-226', 'ip' : ''}]

def generator(inputs, labels):
    #to fit with different input dimension
    #https://github.com/keras-team/keras/issues/1920#issuecomment-410982673
    i = 0
    while True:
        inputs_batch = np.expand_dims([inputs[i%len(inputs)]], axis=2)
        labels_batch = np.array([labels[i%len(inputs)]])
        yield inputs_batch, labels_batch
        i+=1

def model_learning_with_validation(model,X,y,fault_len, class_overweight=1, verbose=0):
    #shuffle the data
    if len(X)==0:
        return model, X, y
    assert len(X)==len(y)
    if fault_len==0:
        class_weight={0:1,1:1}
    else:
        weight_for_0=(1/(len(X)-fault_len))*len(X)
        weight_for_1=(1/(fault_len))*len(X)*class_overweight
        class_weight={0:weight_for_0, 1:weight_for_1}
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
        epochs=1000, verbose=verbose, callbacks=[es,mc],
        class_weight=class_weight)
    return model, X_test, y_test

def model_learning(model,X,y, fault_len,class_overweight=1, verbose=0):
    assert len(X)==len(y)
    if fault_len==0:
        class_weight={0:1,1:1}
    else:
        weight_for_0=(1/(len(X)-fault_len))*len(X)
        weight_for_1=(1/(fault_len))*len(X)*class_overweight
        class_weight={0:weight_for_0, 1:weight_for_1}
    X=np.array(X)
    y=np.array(y)
    es = EarlyStopping(monitor = 'loss', mode = 'min', verbose = 0, patience = 3)
    mc = ModelCheckpoint('best_model.h5', monitor = 'acc', mode = 'max', verbose = 0, save_best_only = True)
    model.fit_generator(generator(X, y), steps_per_epoch=len(X), epochs=1000, verbose=verbose, callbacks=[mc], class_weight=class_weight)
    return model

def CNN_learning(model,gap,win_size):
    test_all_X=[]
    test_all_y=[]
    all_fault_len=0
    today=datetime.today().strftime("%m-%d")
    today=("05-24")
    for vnf in vnf_list:
        vnf_=vnf['vnf_name']
        date_list=date_range('05-11',today, vnf_)
        fault_=get_fault_history(vnf['vnf_num'])
        print('Reading %d th VNF, %s'%(vnf['vnf_num'], vnf_))
        X,y, fault_len=make_data(vnf,fault_,win_size,gap,date_list, over_sampling=10,under_sampling=30)
        model, test_X,test_y=model_learning_with_validation(model, X,y, fault_len, class_overweight=3, verbose=0)
        if len(test_X)>0:
            test_all_X.extend(test_X)
            test_all_y.extend(test_y)
            '''loss, accuracy, f1_score, precision, recall = model.evaluate_generator(
                    generator(test_X, test_y), steps= len(test_X))
            print("original F1 : %.4f, Acc : %.4f, Prec : %.4f, Rec : %.4f"%(f1_score, accuracy, precision, recall))'''
            Model.evaluation(model, test_X, test_y)
        all_fault_len+=fault_len
        model.save("models/CNN_gap%d_win%d"%(gap, win_size))
        print("model saved")
    #loss, accuracy, f1_score, precision, recall = model.evaluate_generator(
    #        generator(test_all_X, test_all_y), steps= len(test_all_X))
    print("-------Learning End. Final performance is here---------")
    print("total test data len is %d and total fault len is %d"%(len(test_all_X),all_fault_len))
    #print("original F1 : %.4f, Acc : %.4f, Prec : %.4f, Rec : %.4f"%(f1_score, accuracy, precision, recall))
    Model.evaluation(model,test_all_X,test_all_y)
    data=[test_all_X,test_all_y]
    with open("models/cnn_data_win%d_gap%d.bin"%(win_size,gap), 'wb') as f:
        pkl.dump(data, f)
    print("test data saved")

def CRNN_learning(model):
    gap=3
    win_size=5
    today=datetime.today().strftime("%m-%d")
    for vnf in vnf_list:
        vnf_=vnf['vnf_name']
        date_list=date_range('05-11',today, vnf_)
        random.shuffle(date_list)
        #Seperate dates as 8:2 to learning and test
        date_learning=date_list[:int(0.8*len(date_list))]
        date_test=date_list[int(0.8*len(date_list)):]
        fault_=get_fault_history(vnf['vnf_num'])
        print('Reading %d th VNF, %s'%(vnf['vnf_num'], vnf_))
        X,y, fault_len= make_data(vnf, fault_, win_size,gap,date_learning,sliding=5,use_emptylog=True, under_sampling=2)
        model=model_learning(model, X,y,fault_len,  verbose=1)
        test_X,test_y, _= make_data(vnf, fault_, win_size,gap,date_test,sliding=5,use_emptylog=True, under_sampling=4)
        if not  len(test_X)==0:
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
    model_file=False
    if not args.model:
        if args.type =='cnn':
            model = Model.CNN()
        elif args.type=='crnn':
            model = Model.CRNN()
    else:
        model = load_model(args.model, custom_objects={"f1_m":Model.f1_m, "precision_m":Model.precision_m, "recall_m":Model.recall_m})
    if args.type =='cnn':
        gap=5
        win_size=10
        print("-------------gap : %d  , win_size : %d Start -------------"%(gap,win_size))
        CNN_learning(model,gap,win_size)
    elif args.type=='crnn':
        CRNN_learning(model)
