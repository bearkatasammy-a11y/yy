import pandas as pd
import numpy as np
from torch.optim import Adam
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
# -*- coding: utf-8 -*-
# 导入库pip install openpyxl -i https://pypi.tuna.tsinghua.edu.cn/simple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error  # 评价指标
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras import optimizers
import keras
import tensorflow as tf
#  mse rmse mae rmape
#  adam sgd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings("ignore")  # 忽略一些警告 不影响运行
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
    squence = 20  # 历史数据的！！！！

    # 预测未来几步？
    pred_len = 2

    train_x = []
    train_y = []
    for i in range(0, len(data) - squence - pred_len, 1):
        train_x.append(data[i:i + squence, :])
        train_y.append(data[i + squence:i + squence + pred_len, 0])
    train_x = np.array(train_x)
    train_x = train_x.reshape(train_x.shape[0], squence, -1)
    train_y = np.array(train_y)
    train_y = train_y.reshape(train_x.shape[0], pred_len)

    print(train_x.shape)
    print(train_y.shape)
    return train_x, train_y
# data_read_csv()

train_x,train_y=data_read_csv()
print(train_x.shape)
print(train_y.shape)
# 序列长度
# int_sequence_len = train_x.shape[1]
# # 每个序列的长度
# int_a = train_x.shape[2]
#
# # 输出几个元素 几步：
# out_len = train_y.shape[1]

# 划分验证集和测试集

x_train, x_test, y_train, y_test = train_test_split(np.array(train_x), np.array(train_y), test_size=0.2, random_state=1)
features = torch.tensor(x_train, dtype=torch.float32)
targets = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(embed_dim, 3*embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size = x.size(0)
        qkv = self.qkv_proj(x).reshape(batch_size, -1, self.num_heads, 3*self.head_dim).transpose(1, 2)
        # print("qkv shape:", qkv.shape)  # 打印 qkv 的形状
        q, k, v = qkv.chunk(3, dim=-1)
        # print("q shape:", q.shape)  # 打印 q 的形状
        q = q / self.head_dim ** 0.5
        scores = (q @ k.transpose(-2, -1))
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x) + residual
        x = self.dropout(x)
        residual = x
        x = self.norm2(x)
        x = self.feedforward(x) + residual
        x = self.dropout(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, feedforward_dim, dropout):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_layers, num_heads, feedforward_dim, dropout):
        super(TransformerModel, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.encoder = TransformerEncoder(embed_dim, num_layers, num_heads, feedforward_dim, dropout)
        self.out_proj = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x) # 99 torch.Size([16, 5, 32])
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.out_proj(x)
        return x


model = TransformerModel(input_dim=18, output_dim=2, embed_dim=32, num_layers=4, num_heads=8, feedforward_dim=64, dropout=0.1)

# 定义损失函数和优化器
from torch.optim import RMSprop
criterion = nn.MSELoss()
optimizer = RMSprop(model.parameters(), lr=0.001)

# 定义训练参数
num_epochs = 100
batch_size = 64

# 训练模型
# 训练模型
train_epoch_loss=[]
for epoch in range(num_epochs):
    temp_loss=[]
    # 数据集划分为小批量
    num_batches = (len(features) + batch_size - 1) // batch_size  # 计算总共有多少个 batch
    with tqdm(total=num_batches, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        for i in range(0, len(features), batch_size):
            batch_features = features[i:i + batch_size]
            batch_targets = targets[i:i + batch_size]

            # 模型前向传播
            outputs = model(batch_features)

            # 计算损失函数
            loss = criterion(outputs, batch_targets)
            temp_loss.append(loss.item())
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update(1)  # 更新进度条
    train_epoch_loss.append(np.mean(temp_loss))
    print('epoch',epoch,'loss',np.mean(temp_loss))
batch_size=1
num_batches = (len(features) + batch_size - 1) // batch_size  # 计算总共有多少个 batch
pred_list=[]
true_list=[]

with torch.no_grad():
    for i in range(0, len(x_test), batch_size):
                batch_features = features[i:i + batch_size]
                batch_targets = targets[i:i + batch_size]
                # 模型前向传播
                outputs = model(batch_features)
                true_list.append(batch_targets.cpu().numpy()[0][0])
                pred_list.append(outputs.cpu().numpy()[0][0])
pred_list=np.array(pred_list)
true_list=np.array(true_list)
from metra import metric
mae, mse, rmse, mape, mspe,r2=metric(pred_list,true_list)
print('mae, mse, rmse, mape, mspe,r2')
print(mae, mse, rmse, mape, mspe,r2)
# 设置Seaborn样式
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
x = range(len(true_list))
data = pd.DataFrame({'x': x, 'y_pred': pred_list.flatten(), 'y_true': true_list.flatten()})
# 绘制y_pred的折线图
sns.lineplot(x='x', y='y_pred', data=data, linewidth=1, label='y_pred')

# 绘制y_true的折线图
sns.lineplot(x='x', y='y_true', data=data, linewidth=1, label='y_true')

# 添加标题和标签
plt.title('Prediction vs True')
plt.xlabel('Date')
plt.ylabel('Values')
plt.savefig('PredictionvsTrue20.png')
# 显示图形
plt.show()



# # 预测结果
# test_features = torch.tensor([[1.5, 1.6, 1.7, 1.8]], dtype=torch.float32)
# predicted_targets = model(test_features)
#
# # 将预测结果转换为DataFrame格式
# predicted_df = pd.DataFrame(predicted_targets.detach().numpy(),
#                             columns=['target1', 'target2', 'target3', 'target4', 'target5', 'target6', 'target7',
#                                      'target8', 'target9', 'target10'])
#
# # 将预测结果写入Excel文件
# predicted_df.to_excel('predicted_results.xlsx', index=False)