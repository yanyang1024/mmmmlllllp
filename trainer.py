"""
模型训练器
负责模型的训练、验证和评估过程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from typing import Dict, List, Optional, Tuple, Callable
from tqdm import tqdm
from loguru import logger

from mlp_model import MLPModel, EarlyStopping, ModelCheckpoint
from data_processor import calculate_metrics


class MLPTrainer:
    """MLP模型训练器类"""
    
    def __init__(self, 
                 model: MLPModel,
                 config: dict,
                 device: Optional[torch.device] = None):
        """
        初始化训练器
        
        Args:
            model: MLP模型
            config: 配置字典
            device: 计算设备
        """
        self.model = model
        self.config = config
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 将模型移动到指定设备
        self.model.to(self.device)
        
        # 初始化优化器
        self.optimizer = self._create_optimizer()
        
        # 初始化损失函数
        self.criterion = self._create_loss_function()
        
        # 初始化学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 训练历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        # 早停和检查点
        self.early_stopping = None
        self.model_checkpoint = None
        
        logger.info(f"训练器初始化完成，使用设备: {self.device}")
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        optimizer_config = self.config['optimizer']
        training_config = self.config['training']
        
        optimizer_type = optimizer_config['type'].lower()
        lr = training_config['learning_rate']
        weight_decay = training_config['weight_decay']
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            momentum = optimizer_config.get('momentum', 0.9)
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'rmsprop':
            optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
            
        logger.info(f"创建优化器: {optimizer_type}, 学习率: {lr}")
        return optimizer
        
    def _create_loss_function(self) -> nn.Module:
        """创建损失函数"""
        loss_config = self.config['loss']
        loss_type = loss_config['type'].lower()
        
        if loss_type == 'mse':
            criterion = nn.MSELoss()
        elif loss_type == 'mae':
            criterion = nn.L1Loss()
        elif loss_type == 'huber':
            criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"不支持的损失函数类型: {loss_type}")
            
        logger.info(f"创建损失函数: {loss_type}")
        return criterion
        
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        # 使用ReduceLROnPlateau调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-7
        )
        
        logger.info("创建学习率调度器: ReduceLROnPlateau")
        return scheduler
        
    def setup_callbacks(self, save_dir: str) -> None:
        """设置回调函数"""
        training_config = self.config['training']
        
        # 早停
        if training_config.get('early_stopping_patience', 0) > 0:
            self.early_stopping = EarlyStopping(
                patience=training_config['early_stopping_patience'],
                min_delta=training_config.get('early_stopping_min_delta', 1e-6),
                restore_best_weights=True
            )
            logger.info(f"启用早停机制，耐心值: {training_config['early_stopping_patience']}")
        
        # 模型检查点
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, 'best_model.pth')
        self.model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        logger.info(f"启用模型检查点保存: {checkpoint_path}")
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, dict]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            平均损失和评估指标
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(train_loader, desc="训练", leave=False)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（可选）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            
            # 收集预测结果用于计算指标
            all_predictions.append(output.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        
        # 计算评估指标
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        metrics = calculate_metrics(targets, predictions)
        
        return avg_loss, metrics
        
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, dict]:
        """
        验证一个epoch
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            平均损失和评估指标
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="验证", leave=False)
            
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # 记录损失
                total_loss += loss.item()
                
                # 收集预测结果
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                
                # 更新进度条
                progress_bar.set_postfix({'loss': loss.item()})
        
        # 计算平均损失
        avg_loss = total_loss / len(val_loader)
        
        # 计算评估指标
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        metrics = calculate_metrics(targets, predictions)
        
        return avg_loss, metrics
        
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: Optional[int] = None,
              save_dir: str = "models") -> dict:
        """
        完整的训练过程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数，如果为None则使用配置中的值
            save_dir: 模型保存目录
            
        Returns:
            训练历史记录
        """
        if epochs is None:
            epochs = self.config['training']['epochs']
            
        # 设置回调函数
        self.setup_callbacks(save_dir)
        
        logger.info(f"开始训练，总轮数: {epochs}")
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 训练
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_metrics = self.validate_epoch(val_loader)
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # 计算epoch时间
            epoch_time = time.time() - epoch_start_time
            
            # 打印进度
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}, "
                f"训练R²: {train_metrics['r2']:.4f}, 验证R²: {val_metrics['r2']:.4f}, "
                f"学习率: {self.optimizer.param_groups[0]['lr']:.2e}, "
                f"时间: {epoch_time:.2f}s"
            )
            
            # 保存最佳模型
            if self.model_checkpoint is not None:
                self.model_checkpoint(val_loss, self.model, self.optimizer, epoch)
            
            # 早停检查
            if self.early_stopping is not None:
                if self.early_stopping(val_loss, self.model):
                    logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                    break
        
        total_time = time.time() - start_time
        logger.info(f"训练完成，总时间: {total_time:.2f}s")
        
        return self.history
        
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用训练好的模型进行预测
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            预测结果和真实值
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="预测", leave=False)
            
            for data, target in progress_bar:
                data = data.to(self.device)
                
                # 预测
                output = self.model(data)
                
                # 收集结果
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        return predictions, targets
        
    def predict_single(self, x: np.ndarray) -> np.ndarray:
        """
        对单个样本或批次进行预测
        
        Args:
            x: 输入数据，形状为 (n_samples, n_features) 或 (n_features,)
            
        Returns:
            预测结果
        """
        self.model.eval()
        
        # 确保输入是二维的
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        # 转换为张量
        x_tensor = torch.FloatTensor(x).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(x_tensor)
            
        return predictions.cpu().numpy()
        
    def save_model(self, filepath: str) -> None:
        """
        保存完整的模型状态
        
        Args:
            filepath: 保存路径
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': self.model.get_model_info(),
            'training_config': self.config,
            'history': self.history
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"模型已保存到: {filepath}")
        
    def load_model(self, filepath: str) -> None:
        """
        加载模型状态
        
        Args:
            filepath: 模型文件路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
            
        logger.info(f"模型已从 {filepath} 加载")
        
    def get_training_summary(self) -> dict:
        """获取训练摘要"""
        if not self.history['train_loss']:
            return {"message": "尚未开始训练"}
            
        best_epoch = np.argmin(self.history['val_loss'])
        
        summary = {
            'total_epochs': len(self.history['train_loss']),
            'best_epoch': best_epoch + 1,
            'best_val_loss': self.history['val_loss'][best_epoch],
            'best_val_r2': self.history['val_metrics'][best_epoch]['r2'],
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'final_train_r2': self.history['train_metrics'][-1]['r2'],
            'final_val_r2': self.history['val_metrics'][-1]['r2']
        }
        
        return summary