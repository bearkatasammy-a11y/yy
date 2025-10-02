import pandas as pd
import numpy as np
from torch.optim import Adam
import torch #导入了 PyTorch 库，它是一个流行的开源机器学习库，用于计算图、梯度和神经网络的开发。
import torch.nn as nn #torch.nn是PyTorch中负责构建神经网络的模块。这里将其别名为 nn，以方便后续调用其中的类和函数。
import torch.nn.functional as F #torch.nn.functional 包含一个函数集合，这些函数通常用作神经网络层的操作，如激活函数、损失函数等，这里以 F 作为别名。
from tqdm import tqdm #tqdm 是一个快速、可扩展的Python进度条，可以在Python长循环中使用，以提供进度和剩余时间的可视化指标。
# -*- coding: utf-8 -*-
# 导入库pip install openpyxl -i https://pypi.tuna.tsinghua.edu.cn/simple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#train_test_split 是在 sklearn.model_selection 模块中，用于将数据集划分为训练集和测试集。通过随机地将数据切分成不同的子集来帮助评估模型性能。
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
#设置了Matplotlib的字体参数，用于正确显示中文字符（如’SimHei’字体）以及负号（即将“axes.unicode_minus”设置为False）。
import warnings
warnings.filterwarnings("ignore")  # 通过filterwarnings("ignore")，告诉Python忽略发出的警告消息。
import numpy as np
import pandas as pd
#导入 NumPy 库（用于数值计算）和 Pandas 库（用于数据分析和操作）。
import os
# data = pd.read_excel("时间序列用电量数据-小时级别.xlsx")  # 1 3 7 是 预测列
# data = data.fillna(-1)
# print(data.columns) # Index(['数据采集时间', '每小时的用电量'], dtype='object')
# print(data.values)


def data_read_csv():
    data = pd.read_excel("data_set.xlsx")  # 1 3 7 是 预测列
    #data_read_csv 函数开始读取一个名为"data_set.xlsx"的Excel文件并将其存储在data变量中。
    data = data.fillna(0)#用0替换了DataFrame中的任何缺失值（即NaN值）
    print(data.columns)
    print(data.values)#打印DataFrame的列名以及它的值，主要是作为一个检查，以确认数据被加载和转换成期望的格式。
    data = data[['count', 'season', 'holiday', 'workingday', 'weekend',
                 'hourpollutant', 'aqi', 'airquality', 'weather', 'pressure', 'temp',
                 'humidity', 'windspeed', 'windtrend', 'windpower', 'visibility',
                 'music', 'railway']].values
    #该行代码将data DataFrame 缩减到指定的列，并转换为NumPy数组格式。
    squence = 10#考虑的历史时间步长
 # “squence”：这个参数用于定义模型应该考虑多长时间的历史数据。例如，如果 “squence” 设为 10，那么在进行预测时，模型将会考虑前 10 个时间步的数据。
    # 如果 “squence” 取值较小，可能会忽视一些长期的历史模式或趋势。相反，如果取值过大，可能会使模型必须处理更多的数据，增加了复杂性，且可能引入不必要的噪声或模式。
    pred_len = 2#预测未来的时间步长
#“pred_len”：这个参数用于确定期望预测未来多少时间步的数据。例如，如果 “pred_len” 设为 2，那么模型将会尝试预测从当前时间点开始的未来两个时间步的数据。
    # 如果 “pred_len” 取值较小，可能只能提供短视的预测，这在需要长期规划的场合可能不够。然而，如果取值过大，预测的不确定性可能会增加，因为预测更远的未来通常更难。
    train_x = []
    train_y = []
    for i in range(0, len(data) - squence - pred_len, 1):
        train_x.append(data[i:i + squence, :])
        train_y.append(data[i + squence:i + squence + pred_len, 0])
    # 通过循环构建时间序列数据集，根据定义的历史时间步长和预测时间步长来切片原始数据，生成训练集train_x和训练标签train_y。
    train_x = np.array(train_x)
    train_x = train_x.reshape(train_x.shape[0], squence, -1)
    train_y = np.array(train_y)
    train_y = train_y.reshape(train_x.shape[0], pred_len)
#这些行将生成的列表转换为NumPy数组，并对train_x和train_y重新整形，使其符合后续模型训练所需的形状。
    print(train_x.shape)
    print(train_y.shape)
    #打印出train_x和train_y的形状，了解数据集的维度。
    return train_x, train_y
train_x,train_y=data_read_csv()
#函数返回整理好的训练数据和标签，并且调用函数以获取这些数据。培训数据和标签被赋值到train_x和train_y变量。
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
#利用train_test_split函数将数据分为训练集和测试集，测试集占整个数据集的20%，设置随机种子为1以确保结果的可重复性。
features = torch.tensor(x_train, dtype=torch.float32)
targets = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
#将NumPy数组转换为PyTorch张量，并且确保它们的类型为float32，这准备了数据用于PyTorch模型的训练和评估。
class MultiHeadAttention(nn.Module):
    #定义了一个名为 MultiHeadAttention 的类，继承自 PyTorch 的 nn.Module，也就是一个可以在神经网络中使用的模块。
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        #这是类的初始化方法。它接受两个参数：embed_dim（输入向量的维度）和num_heads（要使用的注意力头的数量）。
        # super(...).__init__()是一个特殊的调用，用来初始化继承自 nn.Module 的基础构造函数。
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        #这里设置了一些类的属性。embed_dim 是每个输入向量的维度；num_heads 是分割计算的注意力机制的“头”的数量；
        # head_dim 是每个头处理的向量分量的维度，通过总维度除以头的数量得到。
        self.qkv_proj = nn.Linear(embed_dim, 3*embed_dim)
        #这个线性层 (全连接层) 将用于投影输入以生成查询 (query)，键 (key) 和值 (value) 向量。
        # 输出的维度是 3*embed_dim，因为它将同时用于三者。
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        #这是输出层，它将多头注意力的输出再次投影回原始 embed_dim 维度。

    def forward(self, x):#定义了前向传播函数，它接受输入张量 x。
        batch_size = x.size(0)#获取输入的批量大小 batch_size。
        qkv = self.qkv_proj(x).reshape(batch_size, -1, self.num_heads, 3*self.head_dim).transpose(1, 2)
        #应用上面定义的 qkv_proj 层，并整形和变换张量以准备对查询，键和值进行分割。
        # print("qkv shape:", qkv.shape)  # 打印 qkv 的形状
        q, k, v = qkv.chunk(3, dim=-1)
        #使用 chunk 方法把查询，键和值在最后一个维度(dim=-1)切分为三部分。
        # print("q shape:", q.shape)  # 打印 q 的形状
        q = q / self.head_dim ** 0.5
        #对查询进行缩放，这是注意力计算的一部分，有助于训练的稳定性。
        scores = (q @ k.transpose(-2, -1))
        #计算注意力分数，查询和键的转置乘法。
        attn_weights = F.softmax(scores, dim=-1)
        #使用 softmax 函数归一化分数，以得到注意力权重。
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        #通过矩阵乘法应用注意力权重到值上，之后转置以及重新整形以得到多头注意力的输出。
        attn_output = self.out_proj(attn_output)
        #应用输出层到多头注意力的输出，以得到最终的输出张量 attn_output。
        return attn_output#返回最终的注意力输出。这就结束了 MultiHeadAttention 类的逻辑。

class TransformerEncoderLayer(nn.Module):#这是编码器层的PyTorch模块，编码器层是 Transformer 架构的组成部分。
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout):
        super(TransformerEncoderLayer, self).__init__()
        #初始化函数，它接受以下参数：embed_dim（输入向量维度），num_heads（多头注意力中"头"的数量），
        # feedforward_dim（前馈神经网络的隐藏层维度），以及dropout（用于正则化的dropout率）。
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        #创建一个 MultiHeadAttention 类的实例，负责执行多头自注意力机制.
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(dropout),
        )#Sequential container 包含一个前馈神经网络，它有两个线性层和中间的ReLU激活函数，最后是Dropout层。
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        #创建两个层归一化层，用于规范化每个子层之前的输入
        self.dropout = nn.Dropout(dropout)
        #定义一个 Dropout 层，它将在多头注意力输出和前馈神经网络输出之后应用。

    def forward(self, x):#定义前向传播函数，接受输入 x。
        residual = x#存储残差连接前的输入，用于之后的残差连接（实现 skip connection）。
        x = self.norm1(x)
        x = self.self_attn(x) + residual
        x = self.dropout(x)
        #应用层归一化，然后通过自注意力层，最后将其结果加上残差连接，通过 Dropout 层。
        residual = x#更新残差连接的值，用于下一个残差连接。
        x = self.norm2(x)
        x = self.feedforward(x) + residual
        x = self.dropout(x)
        #再次应用层归一化，并通过前馈神经网络，加上第二个残差连接，然后通过 Dropout 层。
        return x
    #返回编码器层的输出。这就完成了 TransformerEncoderLayer 类的介绍。

class TransformerEncoder(nn.Module):
    #定义了一个名为 TransformerEncoder 的类，这个类继承自 PyTorch 的 nn.Module，表示这是一个可用于神经网络的模块
    def __init__(self, embed_dim, num_layers, num_heads, feedforward_dim, dropout):
        super(TransformerEncoder, self).__init__()
        #初始化函数，接受以下参数：embed_dim（输入向量维度），num_layers（编码器数量），
        # num_heads（每个 MultiHeadAttention 中“头”的数量），feedforward_dim（前馈网络的维度），以及 dropout（用于正则化的 dropout 率）。
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        ])#创建一个 ModuleList，里面包含了 num_layers 层的 TransformerEncoderLayer。
        # 每个 TransformerEncoderLayer 都使用相同的参数初始化。
        self.norm = nn.LayerNorm(embed_dim)
        #定义了一个层归一化（Layer Normalization），这是应用于最后编码器输出的归一化层。

    def forward(self, x):#定义了前向传播函数，它接受输入张量 x。
        for layer in self.layers:
            x = layer(x)
            #遍历 ModuleList 中的每一个编码器层，并顺序地应用到输入数据 x。
        x = self.norm(x)
        #对最后一层编码器的输出应用层归一化。
        return x
    #返回编码器处理后的输出。这结束了 TransformerEncoder 类的定义。

class TransformerModel(nn.Module):#定义了一个名为 TransformerModel 的类，这个类同样继承自 PyTorch 的 nn.Module。
    def __init__(self, input_dim, output_dim, embed_dim, num_layers, num_heads, feedforward_dim, dropout):
        super(TransformerModel, self).__init__()
        #这是 TransformerModel 的初始化函数，它会传入模型构造所需要的参数。
        self.embed_dim = embed_dim
        #设置一个类属性以存储嵌入(embedding)的维度。
        self.embedding = nn.Linear(input_dim, embed_dim)
        #定义一个线性层，其目的是将输入的维度映射到嵌入维度上
        self.encoder = TransformerEncoder(embed_dim, num_layers, num_heads, feedforward_dim, dropout)
        #实例化 TransformerEncoder 类，用于创建 Transformer 编码器堆栈。
        self.out_proj = nn.Linear(embed_dim, output_dim)
        #定义一个输出层，其目的是将编码器堆栈的输出维度映射到最终所需的输出维度上。

    def forward(self, x):#定义前向传播函数，它接受输入 x。
        x = self.embedding(x) # 99 torch.Size([16, 5, 32])
        #通过嵌入层将输入 x 映射到嵌入空间。
        x = self.encoder(x)
        #将嵌入向量传送至 Transformer 编码器。
        x = x.mean(dim=1)
        #在序列的时间维度上对编码器的输出取均值，这将把序列转换为单个向量，这在某些任务(例如分类任务)中是常见的做法
        x = self.out_proj(x)
        #将压缩后的编码器输出通过输出层映射到最终的输出维度上。
        return x # 返回模型的最终输出。


model = TransformerModel(input_dim=18, output_dim=2, embed_dim=32, num_layers=4, num_heads=8, feedforward_dim=64, dropout=0.1)
#初始化 TransformerModel 类的一个实例，设置了输入维度、输出维度、嵌入维度、编码器层数、注意力头数、前馈网络维度以及 dropout 率。
# 定义损失函数和优化器
from torch.optim import RMSprop
#引入 RMSprop 优化器，这是一种常用的梯度下降变种。
criterion = nn.MSELoss()
#定义损失函数，使用均方误差损失（Mean Squared Error Loss），这通常在回归问题中使用。
optimizer = RMSprop(model.parameters(), lr=0.001)
#实例化优化器并设置其参数。它将优化模型的参数，并设置了学习率 lr=0.001。

# 定义训练参数
num_epochs = 100
batch_size = 64
#定义训练时将使用的迭代次数（epochs）和用于每次梯度更新的样本集大小（batch size）。

# 训练模型
train_epoch_loss=[]#定义了一个列表train_epoch_loss来收集每个epoch的平均损失。
for epoch in range(num_epochs):#使用一个循环来进行num_epochs次迭代。
    temp_loss=[]#每次epoch开始，初始化一个空列表temp_loss来收集该epoch内所有批次的损失。
    # 数据集划分为小批量
    num_batches = (len(features) + batch_size - 1) // batch_size  # 计算总共有多少个 batch
    #计算了要处理的批次数量。这里用到一个小技巧，通过 + batch_size - 1 确保即使数据无法被批次大小完美地整除也可以包含所有的数据。
    with tqdm(total=num_batches, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        #用tqdm进度条来可视化训练过程。desc为进度条描述，total设置总步数。
        for i in range(0, len(features), batch_size):
            #遍历整个数据集，每次增加batch_size数量的索引。
            batch_features = features[i:i + batch_size]
            batch_targets = targets[i:i + batch_size]
            #切片操作获取当前批次的特征和目标。

            # 模型前向传播
            outputs = model(batch_features)
            #正向传播：将特征输入模型得到预测结果。

            # 计算损失函数
            loss = criterion(outputs, batch_targets)
            #计算预测结果和真实值之间的损失。

            temp_loss.append(loss.item())
            #将当前批次的损失添加到temp_loss列表中。

            # 反向传播和优化
            optimizer.zero_grad()#清除之前的梯度信息。
            loss.backward()#反向传播：根据损失计算梯度。
            optimizer.step()#优化器根据计算得到的梯度更新模型的参数。

            pbar.update(1)  # 更新进度条
    train_epoch_loss.append(np.mean(temp_loss))
    #计算该epoch内所有批次损失的平均值，并添加到train_epoch_loss中。
    print('epoch',epoch,'loss',np.mean(temp_loss))
#打印当前epoch数和平均损失。之后，代码片段中修改了batch_size为1，这通常是为了进行单个样本的预测。
batch_size=1
num_batches = (len(features) + batch_size - 1) // batch_size  # 计算总共有多少个 batch
#重新计算批次数量用于预测，每个批次只有一个样本。然后，代码似乎准备进行一轮的预测或测试：
pred_list=[]
true_list=[]
#两个空列表pred_list和true_list被初始化用来存储预测和真实标签。
with torch.no_grad():
    #使用PyTorch的torch.no_grad()上下文管理器，这意味着接下来的代码块在执行推理时不会计算梯度，从而节省了计算资源并降低了内存消耗，
    # 这在模型评估阶段是常见的做法。
    for i in range(0, len(x_test), batch_size):
        #开始一个循环，用来遍历测试数据集。由于之前设置了batch_size=1，因此这个循环就是单样本推理。
                batch_features = features[i:i + batch_size]
                batch_targets = targets[i:i + batch_size]
        #从features和targets中分别截取当前批次的数据和标签。

                # 模型前向传播
                outputs = model(batch_features)
        #对当前批次的特征进行正向传播，得到预测结果。
                true_list.append(batch_targets.cpu().numpy()[0][0])
                pred_list.append(outputs.cpu().numpy()[0][0])
#将模型输出和真实目标值从GPU转移到CPU，并且转换成NumPy数组格式，然后追加到对应的列表中。这里假设输出和目标是批量大小为1的，因此取了[0][0]。
pred_list=np.array(pred_list)
true_list=np.array(true_list)
#将pred_list和true_list转换为NumPy数组，以便进行后续的计算和可视化处理。
from metra import metric
mae, mse, rmse, mape, mspe,r2=metric(pred_list,true_list)
print('mae, mse, rmse, mape, mspe,r2')
print(mae, mse, rmse, mape, mspe,r2)
# 设置Seaborn样式
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid") #引入seaborn和matplotlib.pyplot用来进行数据可视化，并设置了Seaborn样式为’darkgrid’。
x = range(len(true_list)) #生成一个与true_list长度相同的整数序列，用作图表的x轴。
data = pd.DataFrame({'x': x, 'y_pred': pred_list.flatten(), 'y_true': true_list.flatten()})
#构建一个DataFrame，用于存放位置x、预测值y_pred和真实值y_true。
# 绘制y_pred的折线图
sns.lineplot(x='x', y='y_pred', data=data, linewidth=1, label='y_pred')
# 绘制y_true的折线图
sns.lineplot(x='x', y='y_true', data=data, linewidth=1, label='y_true')
#通过seaborn.lineplot绘制预测值和真实值的折线图

# 添加标题和标签
plt.title('Prediction vs True')
plt.xlabel('Date')
plt.ylabel('Values')
plt.savefig('PredictionvsTrue10.png')
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