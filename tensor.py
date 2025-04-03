#设置 tensor
import numpy as np
from typing import Union, Tuple, Optional

np.random.seed(0)

class Tensor:
    def __init__(self, shape: Union[Tuple[int, ...], int]):
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        self.weight = np.zeros(shape=self.shape, dtype=np.float32)
        self.grad = np.zeros(shape=self.shape, dtype=np.float32)
    
    def grad_clear(self) -> None:
        """清除梯度"""
        self.grad = np.zeros(shape=self.weight.shape, dtype=np.float32)
    
    def __str__(self) -> str:
        return f"Tensor(shape:{self.shape}, weight:{self.weight})"
    
    def __add__(self, other: 'Tensor') -> 'Tensor':
        """张量加法"""
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        result = Tensor(self.shape)
        result.weight = self.weight + other.weight
        return result
    
    def __mul__(self, other: Union['Tensor', float]) -> 'Tensor':
        """张量乘法"""
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
            result = Tensor(self.shape)
            result.weight = self.weight * other.weight
        else:
            result = Tensor(self.shape)
            result.weight = self.weight * other
        return result

class Weight(Tensor):
    def __init__(self, shape: Union[Tuple[int, ...], int], init_method: str = 'normal'):
        super().__init__(shape)
        if init_method == 'normal':
            self.init_weight_normal()
        elif init_method == 'zeros':
            pass  # 使用父类的零初始化
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")

    def init_weight_normal(self, mean: float = 0, std: float = 1) -> None:
        """使用正态分布初始化权重"""
        self.weight = np.random.normal(mean, std, size=self.shape)
        
    def update(self, learning_rate: float) -> None:
        """更新权重"""
        self.weight -= learning_rate * self.grad
        self.grad_clear()
        