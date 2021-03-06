from tqdm import trange
import string
import pickle as pkl
from string import digits
from datetime import datetime, timedelta
from influxdb import DataFrameClient
import numpy as np
import pandas as pd
import yaml
import scp
import paramiko
from scp import SCPClient, SCPException
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import os
import re
import random

local_path='/home/dpnm/tmp/'
remote_path='/mnt/hdd/log/'

def make_data(vnf, fault_, win_size, gap, date_list, sliding=1, use_emptylog=False, over_sampling=1, under_sampling=0, pre_fault_size=5,pre_fault_value=0.65):
    wv= Word2Vec.load('embedding_with_log')
    embed_size=100
    #oov=np.array([1/embed_size**0.5]*embed_size)
    max_fil_size=win_size//2+3
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
        path_=remote_path+vnf['vnf_name']+'/'+log_dir+'/'
        log_file_list=get_file_list(path_)
        log_corpus=[]
        for file_name in log_file_list:
            if vnf['vnf_id']=='server':
                if file_name in ['sudo.log', 'CRON.log', 'nova-compute.log', 'neutron-openvswitch-agent.log', 'apache-access.log']:
                    continue
            elif file_name in ['sudo.log', 'CRON.log', 'stress-ng.log']:
                continue
            log = download_log(path_,file_name,local_path)
            if file_name in ['kernel.log', 'ovs-vswitchd.log']:
                log_token = pre_process_error_only(log)
            else:
                log_token=pre_process(log)
            log_corpus.extend(log_token)

        #Delete Same log. This could be dangerous.
        log_corpus=sorted(list(set(log_corpus)), key= lambda x : x[0])
        if fault_end_date:
            date=datetime.strptime(fault_end_date,'%b-%d-%H:%M')
            fault_end_date=None
        else:
            date=datetime.strptime(log_dir,'%m-%d')
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
                        #pass #Do not add OOV
                        #input_log.extend(np.zeros(embed_size))
                        input_log.extend(np.random.rand(embed_size))
            if use_emptylog and len(input_log) ==0:
                input_log=np.zeros((win_size)*embed_size)
            assert len(input_log)%embed_size==0
            abnormal=False
            fault=False
            for i in range(sliding):
                if (date+timedelta(minutes=gap+win_size+i)).strftime('%b-%d-%H:%M') in fault_['abnormal']:
                    abnormal=True
                    break
                for fault_range in fault_['fault']:
                    if (date+timedelta(minutes=gap+win_size+i)).strftime('%b-%d-%H:%M') == fault_range['start']:
                        fault=True
                        if 'end' in fault_range:
                            fault_end_date=fault_range['end']
                        break
                if fault:
                    break
            if not len(input_log)==0:
                if len(input_log)<max_fil_size*embed_size:
                    input_log.extend(np.zeros(max_fil_size*embed_size-len(input_log)))
                if fault or abnormal:
                    #Not seperate right now.
                    #TODO: seperate learning fault and abnormal
                    y.extend([1]*over_sampling)
                    X.extend([input_log for _ in range(over_sampling)]) #For over Sampling
                    fault_len+=over_sampling
                    if fault:
                        if fault_end_date:
                            if datetime.strptime(log_dir,'%m-%d') < datetime.strptime(fault_end_date[:fault_end_date.find('-',4)],'%b-%d'):
                                #Go to next date.
                                break
                            else:
                                date=datetime.strptime(fault_end_date,'%b-%d-%H:%M')
                                fault_end_date=None
                                continue
                        else:
                            #print("Read total %d number of data and %d of them are fault"%(len(X), fault_len))
                            return X,y, fault_len
                    date+=timedelta(minutes=gap+win_size+sliding-1) ## Slide to after of abnormal
                    continue
                else:
                    #Pr-faults tagging
                    abnormal=False
                    fault=False
                    for i in range(pre_fault_size):
                        if (date+timedelta(minutes=gap+win_size+sliding+i)).strftime('%b-%d-%H:%M') in fault_['abnormal']:
                            abnormal=True
                            break
                        for fault_range in fault_['fault']:
                            if (date+timedelta(minutes=gap+win_size+sliding+i)).strftime('%b-%d-%H:%M') == fault_range['start']:
                                fault=True
                                break
                        if fault:
                            break
                    if abnormal or fault:
                        X.append(input_log)
                        y.append(pre_fault_value)
                    elif random.choice([True]+[False for i in range(under_sampling)]):
                        X.append(input_log)
                        y.append(0)
            date+=timedelta(minutes=sliding)
    #print("Read total %d number of data and %d of them are fault"%(len(X), fault_len))
    return X,y, fault_len

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
    translator = str.maketrans(string.punctuation, ' '*(len(string.punctuation)))
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
    translator = str.maketrans(string.punctuation, ' '*(len(string.punctuation)))
    clean =[]
    for each in text.split('\n'):
        each=each.lower()
        for error_word in ['error', 'reset', 'fail', 'not', 'fault']:
            if error_word in each:
                #Todo : How to use number information??
                clean.append(tuple([datetime.strptime(each[:12], "%b %d %H:%M"), tuple([x for x in  re.sub(r'[0-9]+',
                    '', each[each.find(':',19)+1:]).translate(translator).split() ])]))
    return clean

def download_log(remote_path, file_name, local_path):
    with open ('../server_info.yaml') as f:
        server_info=yaml.load(f)['log']
    try:
        cli=paramiko.SSHClient()
        cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        cli.connect(server_info['ip'], port=22, username=server_info['id'], password=server_info['pwd'])
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
    with open ('../server_info.yaml') as f:
        server_info=yaml.load(f)['log']
    cli=paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    cli.connect(server_info['ip'], port=22, username=server_info['id'], password=server_info['pwd'])
    stdin, stdout, stderr = cli.exec_command('ls '+path)
    rslt=stdout.read()
    file_list=rslt.split()
    del stdin, stdout, stderr
    file_list = [file_name.decode('utf-8') for file_name in file_list]
    return file_list

def date_range(start, end, vnf_name):
    start=datetime.strptime(start, "%m-%d")
    end = datetime.strptime(end, "%m-%d")
    dates = [(start + timedelta(days=i)).strftime("%m-%d") for i in range((end-start).days+1)]
    log_dir_list=get_file_list(remote_path+vnf_name+'/')
    for date in dates[:]:
        if date not in log_dir_list:
            dates.remove(date)
    return dates

def fault_tagging(vnf_num):
    #Tagging based on Packet Processing Time
    #Not Use
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
    #print("Get Fault history")
    fault_history={}
    with open ('../server_info.yaml') as f:
        server_info=yaml.load(f)['FaultHistory']
    cli=paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    cli.connect(server_info['ip'], port=22, username=server_info['id'], password=server_info['pwd'])
    try:
        with SCPClient(cli.get_transport()) as scp:
            scp.get('/home/ubuntu/fault_history.yaml',local_path)
    except SCPException:
        raise SCPException.message
    with open(local_path+'fault_history.yaml')as f:
        fault_history=yaml.load(f)
    return fault_history[vnf_num]
