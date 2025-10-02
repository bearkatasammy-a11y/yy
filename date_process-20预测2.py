import numpy as np
import pandas as pd
import os
# data = pd.read_excel("时间序列用电量数据-小时级别.xlsx")  # 1 3 7 是 预测列
# data = data.fillna(-1)
# print(data.columns) # Index(['数据采集时间', '每小时的用电量'], dtype='object')
# print(data.values)


def data_read_csv():
       data = pd.read_excel("data_set.xlsx")  # 1 3 7 是 预测列
       data = data.fillna(0)
       print(data.columns)
       print(data.values)
       data = data[['count', 'season', 'holiday', 'workingday', 'weekend',
                      'hourpollutant', 'aqi', 'airquality', 'weather', 'pressure', 'temp',
                      'humidity', 'windspeed', 'windtrend', 'windpower', 'visibility',
                      'music', 'railway']].values
       squence=20 # 历史数据的！！！！

       # 预测未来几步？
       pred_len=2

       train_x=[]
       train_y=[]
       for i in range(0,len(data)-squence-pred_len,1):
              train_x.append(data[i:i+squence,:])
              train_y.append(data[i+squence:i+squence+pred_len,0])
       train_x=np.array(train_x)
       train_x=train_x.reshape(train_x.shape[0],squence,-1)
       train_y=np.array(train_y)
       train_y=train_y.reshape(train_x.shape[0],pred_len)

       print(train_x.shape)
       print(train_y.shape)
       return train_x,train_y
data_read_csv()
