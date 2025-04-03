'''设置激活函数、层函数
'''

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
                 in_channels, #输入长度
                 out_channels, #输出类目
                 name = 'linear',
                 bias = True,
                 ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Parameter = Weight(shape = (self.in_channels, self.out_channels), init_method='normal')
        self.bias = np.random.random(self.out_channels) if bias else None
        
    def forward(self, x): #前向传播 x的大小为[,in_channels]
        output_tensor = x @ self.Parameter.weight + self.bias if self.bias is not None else x @ self.Parameter.weight
        return output_tensor


if __name__ == "__main__":
    ln = Linear(4, 2, bias=True)
    print(ln.bias)
    input_list = [[1.0, 2.0, 3.0, 4.0]]
    output = ln(input_list)
    print(output)
