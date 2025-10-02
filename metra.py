import numpy as np
#导入了 NumPy 库，这是一个广泛用于高性能科学计算和数据分析的库。

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))
#这个函数计算预测值 pred 和真实值 true 之间的相对平方误差。返回的是真实值偏差的平方和的平方根与预测误差的平方和的平方根之比。

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)
#这个函数计算预测值与真实值之间的相关性。首先计算分子 u，即每对真实值和预测值差值的乘积之和。计算分母 d，即真实值和预测值差值的平方乘积之和的平方根。最后，返回这些比值的均值

def MAE(pred, true):
    return np.mean(np.abs(pred - true))
#这个函数返回预测值与真实值之差的绝对值的平均值，这是度量预测精度的一种标准方法。

def MSE(pred, true):
    return np.mean((pred - true) ** 2)
#与MAE类似，这个函数返回预测值和真实值之差的平方的平均值。MSE 的开平方即为 RMSE。

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))
#这个函数计算预测误差的均方根，即 MSE 的平方根。

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))
#这个函数计算预测值和真实值之间的绝对百分比误差的平均值

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))
#类似于 MAPE, 计算预测与真实值的百分比误差的平方的平均值。

from sklearn.metrics import r2_score
#这行代码从 sklearn.metrics 模块导入 r2_score 函数，用来计算预测值的决定系数。
def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    r2=r2_score(pred, true)
    return mae, mse, rmse, mape, mspe,r2
#这个 metric 函数将以上定义的度量函数封装在内，分别计算 MAE, MSE, RMSE, MAPE, MSPE 和 R2 分数，然后将它们作为元组返回。
# 注意，这里计算 R2 分数时使用了 r2_score 函数，并且 r2 的参数顺序是先预测值 pred 后真实值 true，而不是通常的顺序，这可能会导致不准确的 R2 分数。