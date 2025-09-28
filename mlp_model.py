"""
多层感知机(MLP)模型实现
用于数值预测任务的深度学习模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
import numpy as np
from loguru import logger


class MLPModel(nn.Module):
    """
    多层感知机模型类
    支持可配置的隐藏层、激活函数和Dropout
    """
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_layers: List[int] = [128, 64, 32],
                 activation: str = 'relu',
                 dropout_rate: float = 0.2,
                 batch_norm: bool = False):
        """
        初始化MLP模型
        
        Args:
            input_dim: 输入特征维度
            output_dim: 输出维度
            hidden_layers: 隐藏层神经元数量列表
            activation: 激活函数类型 ('relu', 'tanh', 'sigmoid', 'leaky_relu')
            dropout_rate: Dropout比率
            batch_norm: 是否使用批标准化
        """
        super(MLPModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        
        # 构建网络层
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        self.dropouts = nn.ModuleList()
        
        # 输入层到第一个隐藏层
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_layers):
            # 线性层
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # 批标准化层
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
            # Dropout层
            self.dropouts.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # 输出层
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # 初始化权重
        self._initialize_weights()
        
        logger.info(f"MLP模型初始化完成: 输入维度={input_dim}, 输出维度={output_dim}, "
                   f"隐藏层={hidden_layers}, 激活函数={activation}")
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Xavier初始化
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        
        # 输出层使用较小的初始化
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        nn.init.constant_(self.output_layer.bias, 0)
    
    def _get_activation(self, x: torch.Tensor) -> torch.Tensor:
        """获取激活函数"""
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'tanh':
            return torch.tanh(x)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation == 'leaky_relu':
            return F.leaky_relu(x, negative_slope=0.01)
        else:
            raise ValueError(f"不支持的激活函数: {self.activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, input_dim)
            
        Returns:
            输出张量，形状为 (batch_size, output_dim)
        """
        # 通过隐藏层
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # 批标准化
            if self.batch_norm and self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            # 激活函数
            x = self._get_activation(x)
            
            # Dropout
            x = self.dropouts[i](x)
        
        # 输出层（不使用激活函数，因为是回归任务）
        x = self.output_layer(x)
        
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测函数，自动处理评估模式
        
        Args:
            x: 输入张量
            
        Returns:
            预测结果
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm
        }


class EarlyStopping:
    """早停机制类"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6, 
                 restore_best_weights: bool = True):
        """
        初始化早停机制
        
        Args:
            patience: 容忍的epoch数量
            min_delta: 最小改善量
            restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        检查是否应该早停
        
        Args:
            val_loss: 验证损失
            model: 模型
            
        Returns:
            是否应该早停
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                import copy
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info("恢复最佳权重")
            return True
            
        return False


class ModelCheckpoint:
    """模型检查点保存类"""
    
    def __init__(self, filepath: str, monitor: str = 'val_loss', 
                 save_best_only: bool = True, mode: str = 'min'):
        """
        初始化模型检查点
        
        Args:
            filepath: 保存路径
            monitor: 监控的指标
            save_best_only: 是否只保存最佳模型
            mode: 'min' 或 'max'
        """
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        
        if mode == 'min':
            self.best_score = float('inf')
            self.is_better = lambda current, best: current < best
        else:
            self.best_score = float('-inf')
            self.is_better = lambda current, best: current > best
    
    def __call__(self, current_score: float, model: nn.Module, 
                 optimizer: torch.optim.Optimizer, epoch: int) -> bool:
        """
        检查并保存模型
        
        Args:
            current_score: 当前分数
            model: 模型
            optimizer: 优化器
            epoch: 当前epoch
            
        Returns:
            是否保存了模型
        """
        if not self.save_best_only or self.is_better(current_score, self.best_score):
            self.best_score = current_score
            
            # 保存模型状态
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': self.best_score,
                'model_config': model.get_model_info()
            }
            
            torch.save(checkpoint, self.filepath)
            logger.info(f"模型已保存到 {self.filepath}, {self.monitor}={current_score:.6f}")
            return True
            
        return False


def create_model_from_config(config: dict, input_dim: int, output_dim: int) -> MLPModel:
    """
    根据配置创建模型
    
    Args:
        config: 配置字典
        input_dim: 输入维度
        output_dim: 输出维度
        
    Returns:
        MLP模型实例
    """
    model_config = config['model']
    
    model = MLPModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=model_config['hidden_layers'],
        activation=model_config['activation'],
        dropout_rate=model_config['dropout_rate'],
        batch_norm=model_config.get('batch_norm', False)
    )
    
    return model


def load_model_checkpoint(filepath: str, model: MLPModel, 
                         optimizer: Optional[torch.optim.Optimizer] = None) -> dict:
    """
    加载模型检查点
    
    Args:
        filepath: 检查点文件路径
        model: 模型实例
        optimizer: 优化器实例（可选）
        
    Returns:
        检查点信息字典
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"模型检查点已加载: {filepath}")
    
    return checkpoint


def count_parameters(model: nn.Module) -> tuple:
    """
    计算模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        (总参数数, 可训练参数数)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params