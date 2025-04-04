'''设置激活函数、层函数
'''

from tkinter import Widget
from flask.cli import F
from  tensor import Weight
import numpy as np


def cross_loss(y_true, y_pred): #计算预算中两个向量的交叉熵
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10) #将其限制np数组，并限制上下限
    y_true = np.array(y_true)
    return - np.sum(y_true * np.log(y_pred)) / len(y_true) #*点乘是对位相乘



#定义层的输出格式
class Layer:
    def __init__(self, name='layer', *args, **kwargs):
        self.name = name

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def parameters(self):
        return []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __str__(self):
        return self.name
    
#定义线性层
class Linear(Layer):
    def __init__(self, 
                 in_channels,  # 输入特征数
                 out_channels, # 输出特征数
                 name='linear',
                 bias=True):
        super().__init__(name)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Weight(shape=(self.in_channels, self.out_channels), init_method='normal')
        self.bias = Weight(shape=(self.out_channels,), init_method='normal') if bias else None
        self.input = None
        
    def forward(self, x):  # x: [batch_size, in_channels]
        self.input = x
        output = x @ self.weight.weight  # 或 x @ self.weight.weight.T（取决于weight定义）
        if self.bias is not None:
            output += self.bias.weight  # 广播加法
        return output  # [batch_size, out_channels]
    
    def backward(self, gradient):  # gradient: [batch_size, out_channels] 从前一层传回的梯度
        # 权重的梯度
        self.weight.grad += self.input.T @ gradient  # [in_channels, out_channels]
        
        # 偏置的梯度
        if self.bias is not None:
            self.bias.grad += gradient.sum(dim=0)  # [out_channels]
        
        # 输入的梯度
        input_gradient = gradient @ self.weight.weight.T  # [batch_size, in_channels]
        return input_gradient

# 定义relu激活层
class ReLU(Layer):
    def __init__(self, 
                 name='relu',
                 ):
        super().__init__(name)
    
    def forward(self,x):
        self.input = x
        return np.maximum(x, 0) #逐行比较两个数组，返回大于等于0的数

    def backward(self, gradient):
        gradient[self.input <= 0] = 0
        return gradient
        

if __name__ == "__main__":
    ln = Linear(4, 2, bias=True)
    print(ln.bias)
    input_list = [[1.0, 2.0, 3.0, 4.0]]
    output = ln(input_list)
    print(output)
