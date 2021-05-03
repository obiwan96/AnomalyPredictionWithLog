import pandas as pd
from pandas.io.parsers import TextParser
from lxml.html import parse
from urllib.request import urlopen, Request
from influxdb import DataFrameClient
import yaml

user=None
password=None
host=None
port=8086

def read_InDB(vnf, event):
    global user, password, host, port
    if event == 'ppt':
        dbname='pptmon'
        query='select * from "%d"'%vnf['vnf_num']
    else :
        dbname = 'ni'
        if event=='cpu':
            query='select * from "%s___cpu_usage___value___gauge"'%vnf['vnf_id']
        else:
            query='select * from "%s___memory_free___value___gauge"'%vnf['vnf_id']
    client=DataFrameClient(host,port,user,password,dbname)
    df=client.query(qeury+' where time > now() -1d')
    df=list(df.values())
    df=df[0].tz_convert('Asia/Seoul')
    df.index=df.index.map(lambda x : x.replace(microsecond=0))
    return df


if __name__ == '__main__':
    vnf_list=[
            {'vnf_num' : 1, 'vnf_id' : 'b21b38dc-eee0-4f27-9863-1b600705a126'},
            {'vnf_num' : 2, 'vnf_id' : '3f97a1f5-f766-4b9c-89dd-ef7f8ef9adb7'},
            {'vnf_num' : 3, 'vnf_id' : '65d6314d-c3cc-46d0-9856-67fc95b953f7'},
            {'vnf_num' : 4, 'vnf_id' : '75389a36-510b-41ac-9a19-380b9716c30c'},
            {'vnf_num' : 5, 'vnf_id' : 'cbe5d6e7-a490-4326-aea4-8035dc8b3d46'},
            {'vnf_num' : 6, 'vnf_id' : 'dc5422d7-9e9d-4dc4-a381-5e2bcc986667'}]
    with open ('../server_info.yaml') as f:
        server_info=yaml.load(f)['InDB']
    host=server_info['ip']
    user=server_info['id']
    password=server_info['password']
    for vnf in vnf_list:
        ppt=read_InDB(vnf, 'ppt')
        cpu=read_cpu(vnf,'cpu')
        memory=read_memory(vnf,'memory')
