"""
数据处理和预处理模块
用于处理数值预测任务的输入数据
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Union
import joblib
import os
from loguru import logger


class NumericalDataset(Dataset):
    """PyTorch数据集类，用于数值预测任务"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        初始化数据集
        
        Args:
            X: 输入特征，形状为 (n_samples, n_features)
            y: 目标值，形状为 (n_samples, n_targets)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class DataProcessor:
    """数据处理器类，负责数据的加载、预处理和分割"""
    
    def __init__(self, config: dict):
        """
        初始化数据处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.scaler_X = None
        self.scaler_y = None
        self.input_dim = None
        self.output_dim = None
        
    def load_data_from_arrays(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        从numpy数组加载数据
        
        Args:
            X: 输入特征数组，形状为 (n_samples, n_features)
            y: 目标值数组，形状为 (n_samples, n_targets)
        """
        logger.info(f"加载数据: X形状={X.shape}, y形状={y.shape}")
        
        # 确保数据是二维的
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        self.X_raw = X.astype(np.float32)
        self.y_raw = y.astype(np.float32)
        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1]
        
        logger.info(f"数据加载完成: 输入维度={self.input_dim}, 输出维度={self.output_dim}")
        
    def load_data_from_csv(self, file_path: str, target_columns: list, 
                          feature_columns: Optional[list] = None) -> None:
        """
        从CSV文件加载数据
        
        Args:
            file_path: CSV文件路径
            target_columns: 目标列名列表
            feature_columns: 特征列名列表，如果为None则使用除目标列外的所有列
        """
        logger.info(f"从CSV文件加载数据: {file_path}")
        
        df = pd.read_csv(file_path)
        
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col not in target_columns]
            
        X = df[feature_columns].values.astype(np.float32)
        y = df[target_columns].values.astype(np.float32)
        
        self.load_data_from_arrays(X, y)
        
    def normalize_data(self, scaler_type: str = 'standard') -> None:
        """
        标准化数据
        
        Args:
            scaler_type: 标准化类型，'standard' 或 'minmax'
        """
        if not self.config['data']['normalize']:
            logger.info("跳过数据标准化")
            self.X_processed = self.X_raw.copy()
            self.y_processed = self.y_raw.copy()
            return
            
        logger.info(f"使用{scaler_type}标准化数据")
        
        # 选择标准化器
        if scaler_type == 'standard':
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler_X = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
        else:
            raise ValueError(f"不支持的标准化类型: {scaler_type}")
            
        # 标准化输入特征
        self.X_processed = self.scaler_X.fit_transform(self.X_raw)
        
        # 标准化目标值
        self.y_processed = self.scaler_y.fit_transform(self.y_raw)
        
        logger.info("数据标准化完成")
        
    def split_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                 np.ndarray, np.ndarray, np.ndarray]:
        """
        分割数据为训练集、验证集和测试集
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        config = self.config['data']
        train_ratio = config['train_ratio']
        val_ratio = config['val_ratio']
        test_ratio = config['test_ratio']
        random_seed = config['random_seed']
        
        logger.info(f"分割数据: 训练集={train_ratio}, 验证集={val_ratio}, 测试集={test_ratio}")
        
        # 首先分离出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X_processed, self.y_processed,
            test_size=test_ratio,
            random_state=random_seed
        )
        
        # 然后从剩余数据中分离训练集和验证集
        val_size = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=random_seed
        )
        
        logger.info(f"数据分割完成: 训练集={X_train.shape[0]}, "
                   f"验证集={X_val.shape[0]}, 测试集={X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def create_data_loaders(self, X_train: np.ndarray, X_val: np.ndarray, 
                           X_test: np.ndarray, y_train: np.ndarray, 
                           y_val: np.ndarray, y_test: np.ndarray,
                           batch_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        创建PyTorch数据加载器
        
        Args:
            X_train, X_val, X_test: 特征数据
            y_train, y_val, y_test: 目标数据
            batch_size: 批次大小，如果为None则使用配置中的值
            
        Returns:
            train_loader, val_loader, test_loader
        """
        if batch_size is None:
            batch_size = self.config['training']['batch_size']
            
        # 创建数据集
        train_dataset = NumericalDataset(X_train, y_train)
        val_dataset = NumericalDataset(X_val, y_val)
        test_dataset = NumericalDataset(X_test, y_test)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0  # Windows环境下设置为0避免多进程问题
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"数据加载器创建完成，批次大小={batch_size}")
        
        return train_loader, val_loader, test_loader
        
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        将标准化后的预测结果转换回原始尺度
        
        Args:
            predictions: 标准化后的预测结果
            
        Returns:
            原始尺度的预测结果
        """
        if self.scaler_y is not None:
            return self.scaler_y.inverse_transform(predictions)
        return predictions
        
    def save_scalers(self, save_dir: str) -> None:
        """
        保存标准化器
        
        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if self.scaler_X is not None:
            joblib.dump(self.scaler_X, os.path.join(save_dir, 'scaler_X.pkl'))
            logger.info("输入特征标准化器已保存")
            
        if self.scaler_y is not None:
            joblib.dump(self.scaler_y, os.path.join(save_dir, 'scaler_y.pkl'))
            logger.info("目标值标准化器已保存")
            
    def load_scalers(self, save_dir: str) -> None:
        """
        加载标准化器
        
        Args:
            save_dir: 保存目录
        """
        scaler_X_path = os.path.join(save_dir, 'scaler_X.pkl')
        scaler_y_path = os.path.join(save_dir, 'scaler_y.pkl')
        
        if os.path.exists(scaler_X_path):
            self.scaler_X = joblib.load(scaler_X_path)
            logger.info("输入特征标准化器已加载")
            
        if os.path.exists(scaler_y_path):
            self.scaler_y = joblib.load(scaler_y_path)
            logger.info("目标值标准化器已加载")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    计算回归评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        包含各种评估指标的字典
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    return metrics


def generate_sample_data(n_samples: int = 1000, n_features: int = 10, 
                        n_targets: int = 3, noise: float = 0.1, 
                        random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成示例数据用于测试
    
    Args:
        n_samples: 样本数量
        n_features: 输入特征数量
        n_targets: 目标变量数量
        noise: 噪声水平
        random_seed: 随机种子
        
    Returns:
        X, y: 输入特征和目标值
    """
    np.random.seed(random_seed)
    
    # 生成输入特征
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # 生成权重矩阵
    W = np.random.randn(n_features, n_targets).astype(np.float32)
    
    # 生成目标值（线性关系 + 非线性变换 + 噪声）
    y_linear = X @ W
    y_nonlinear = np.sin(y_linear) + 0.5 * np.cos(y_linear * 2)
    y = y_nonlinear + noise * np.random.randn(n_samples, n_targets).astype(np.float32)
    
    logger.info(f"生成示例数据: {n_samples}个样本, {n_features}个特征, {n_targets}个目标")
    
    return X, y