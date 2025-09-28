"""
模型评估器
提供全面的模型评估功能，包括指标计算和可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import torch
from typing import Dict, List, Tuple, Optional
import os
from loguru import logger

from data_processor import calculate_metrics


class ModelEvaluator:
    """模型评估器类"""
    
    def __init__(self, save_plots: bool = True, plot_dir: str = "plots"):
        """
        初始化评估器
        
        Args:
            save_plots: 是否保存图表
            plot_dir: 图表保存目录
        """
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        
        if save_plots:
            os.makedirs(plot_dir, exist_ok=True)
            
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def evaluate_model(self, 
                      y_true: np.ndarray, 
                      y_pred: np.ndarray,
                      dataset_name: str = "test") -> Dict:
        """
        全面评估模型性能
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            dataset_name: 数据集名称
            
        Returns:
            评估结果字典
        """
        logger.info(f"开始评估 {dataset_name} 数据集")
        
        # 基础指标
        basic_metrics = calculate_metrics(y_true, y_pred)
        
        # 额外指标
        additional_metrics = self._calculate_additional_metrics(y_true, y_pred)
        
        # 合并所有指标
        all_metrics = {**basic_metrics, **additional_metrics}
        
        # 按目标维度计算指标
        if y_true.shape[1] > 1:
            target_metrics = self._calculate_per_target_metrics(y_true, y_pred)
            all_metrics['per_target_metrics'] = target_metrics
        
        # 打印评估结果
        self._print_evaluation_results(all_metrics, dataset_name)
        
        return all_metrics
        
    def _calculate_additional_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """计算额外的评估指标"""
        metrics = {}
        
        # 平均绝对百分比误差 (MAPE)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        metrics['mape'] = mape
        
        # 对称平均绝对百分比误差 (SMAPE)
        smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
        metrics['smape'] = smape
        
        # 最大误差
        max_error = np.max(np.abs(y_true - y_pred))
        metrics['max_error'] = max_error
        
        # 解释方差分数
        explained_var = 1 - np.var(y_true - y_pred) / np.var(y_true)
        metrics['explained_variance'] = explained_var
        
        # 皮尔逊相关系数
        if y_true.shape[1] == 1:
            correlation, p_value = stats.pearsonr(y_true.flatten(), y_pred.flatten())
            metrics['pearson_correlation'] = correlation
            metrics['pearson_p_value'] = p_value
        
        return metrics
        
    def _calculate_per_target_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """计算每个目标变量的指标"""
        n_targets = y_true.shape[1]
        per_target_metrics = {}
        
        for i in range(n_targets):
            target_true = y_true[:, i]
            target_pred = y_pred[:, i]
            
            metrics = calculate_metrics(target_true.reshape(-1, 1), target_pred.reshape(-1, 1))
            per_target_metrics[f'target_{i}'] = metrics
            
        return per_target_metrics
        
    def _print_evaluation_results(self, metrics: Dict, dataset_name: str) -> None:
        """打印评估结果"""
        logger.info(f"\n=== {dataset_name.upper()} 数据集评估结果 ===")
        logger.info(f"均方误差 (MSE): {metrics['mse']:.6f}")
        logger.info(f"均方根误差 (RMSE): {metrics['rmse']:.6f}")
        logger.info(f"平均绝对误差 (MAE): {metrics['mae']:.6f}")
        logger.info(f"决定系数 (R²): {metrics['r2']:.6f}")
        logger.info(f"平均绝对百分比误差 (MAPE): {metrics['mape']:.2f}%")
        logger.info(f"对称MAPE (SMAPE): {metrics['smape']:.2f}%")
        logger.info(f"最大误差: {metrics['max_error']:.6f}")
        logger.info(f"解释方差分数: {metrics['explained_variance']:.6f}")
        
        if 'pearson_correlation' in metrics:
            logger.info(f"皮尔逊相关系数: {metrics['pearson_correlation']:.6f}")
            
    def plot_predictions_vs_actual(self, 
                                  y_true: np.ndarray, 
                                  y_pred: np.ndarray,
                                  dataset_name: str = "test",
                                  target_names: Optional[List[str]] = None) -> None:
        """
        绘制预测值vs真实值散点图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            dataset_name: 数据集名称
            target_names: 目标变量名称列表
        """
        n_targets = y_true.shape[1]
        
        if n_targets == 1:
            # 单目标情况
            plt.figure(figsize=(8, 6))
            plt.scatter(y_true, y_pred, alpha=0.6, s=20)
            
            # 绘制理想线 (y=x)
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想预测线')
            
            plt.xlabel('真实值')
            plt.ylabel('预测值')
            plt.title(f'{dataset_name.capitalize()} 数据集: 预测值 vs 真实值')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 添加R²信息
            r2 = r2_score(y_true, y_pred)
            plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            if self.save_plots:
                plt.savefig(os.path.join(self.plot_dir, f'{dataset_name}_predictions_vs_actual.png'), 
                           dpi=300, bbox_inches='tight')
            plt.show()
            
        else:
            # 多目标情况
            n_cols = min(3, n_targets)
            n_rows = (n_targets + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
            if n_targets == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
                
            for i in range(n_targets):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                
                target_true = y_true[:, i]
                target_pred = y_pred[:, i]
                
                ax.scatter(target_true, target_pred, alpha=0.6, s=20)
                
                # 理想线
                min_val = min(target_true.min(), target_pred.min())
                max_val = max(target_true.max(), target_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                
                target_name = target_names[i] if target_names else f'目标 {i+1}'
                ax.set_xlabel('真实值')
                ax.set_ylabel('预测值')
                ax.set_title(f'{target_name}')
                ax.grid(True, alpha=0.3)
                
                # 添加R²信息
                r2 = r2_score(target_true, target_pred)
                ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 隐藏多余的子图
            for i in range(n_targets, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                if n_rows > 1:
                    axes[row, col].set_visible(False)
                else:
                    axes[col].set_visible(False)
                    
            plt.tight_layout()
            
            if self.save_plots:
                plt.savefig(os.path.join(self.plot_dir, f'{dataset_name}_predictions_vs_actual_multi.png'), 
                           dpi=300, bbox_inches='tight')
            plt.show()
            
    def plot_residuals(self, 
                      y_true: np.ndarray, 
                      y_pred: np.ndarray,
                      dataset_name: str = "test") -> None:
        """
        绘制残差图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            dataset_name: 数据集名称
        """
        residuals = y_true - y_pred
        
        if y_true.shape[1] == 1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # 残差vs预测值
            ax1.scatter(y_pred, residuals, alpha=0.6, s=20)
            ax1.axhline(y=0, color='r', linestyle='--', lw=2)
            ax1.set_xlabel('预测值')
            ax1.set_ylabel('残差')
            ax1.set_title('残差 vs 预测值')
            ax1.grid(True, alpha=0.3)
            
            # 残差直方图
            ax2.hist(residuals.flatten(), bins=30, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('残差')
            ax2.set_ylabel('频次')
            ax2.set_title('残差分布')
            ax2.grid(True, alpha=0.3)
            
            # 添加正态性检验
            _, p_value = stats.normaltest(residuals.flatten())
            ax2.text(0.05, 0.95, f'正态性检验 p值: {p_value:.4f}', 
                    transform=ax2.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
        else:
            # 多目标情况 - 显示总体残差
            residuals_flat = residuals.flatten()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # 残差vs预测值
            ax1.scatter(y_pred.flatten(), residuals_flat, alpha=0.6, s=20)
            ax1.axhline(y=0, color='r', linestyle='--', lw=2)
            ax1.set_xlabel('预测值')
            ax1.set_ylabel('残差')
            ax1.set_title('残差 vs 预测值 (所有目标)')
            ax1.grid(True, alpha=0.3)
            
            # 残差直方图
            ax2.hist(residuals_flat, bins=30, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('残差')
            ax2.set_ylabel('频次')
            ax2.set_title('残差分布 (所有目标)')
            ax2.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(os.path.join(self.plot_dir, f'{dataset_name}_residuals.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_training_history(self, history: Dict) -> None:
        """
        绘制训练历史
        
        Args:
            history: 训练历史字典
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # 损失曲线
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='训练损失')
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='验证损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].set_title('训练和验证损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # R²分数
        train_r2 = [metrics['r2'] for metrics in history['train_metrics']]
        val_r2 = [metrics['r2'] for metrics in history['val_metrics']]
        
        axes[0, 1].plot(epochs, train_r2, 'b-', label='训练R²')
        axes[0, 1].plot(epochs, val_r2, 'r-', label='验证R²')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('R² 分数')
        axes[0, 1].set_title('R² 分数变化')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # RMSE
        train_rmse = [metrics['rmse'] for metrics in history['train_metrics']]
        val_rmse = [metrics['rmse'] for metrics in history['val_metrics']]
        
        axes[1, 0].plot(epochs, train_rmse, 'b-', label='训练RMSE')
        axes[1, 0].plot(epochs, val_rmse, 'r-', label='验证RMSE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].set_title('RMSE变化')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 学习率
        axes[1, 1].plot(epochs, history['learning_rates'], 'g-')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('学习率')
        axes[1, 1].set_title('学习率变化')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(os.path.join(self.plot_dir, 'training_history.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_evaluation_report(self, 
                               train_metrics: Dict,
                               val_metrics: Dict,
                               test_metrics: Dict,
                               model_info: Dict,
                               save_path: str = "evaluation_report.txt") -> None:
        """
        创建详细的评估报告
        
        Args:
            train_metrics: 训练集指标
            val_metrics: 验证集指标
            test_metrics: 测试集指标
            model_info: 模型信息
            save_path: 报告保存路径
        """
        report = []
        report.append("=" * 60)
        report.append("MLP 数值预测模型评估报告")
        report.append("=" * 60)
        report.append("")
        
        # 模型信息
        report.append("模型配置:")
        report.append(f"  输入维度: {model_info['input_dim']}")
        report.append(f"  输出维度: {model_info['output_dim']}")
        report.append(f"  隐藏层: {model_info['hidden_layers']}")
        report.append(f"  激活函数: {model_info['activation']}")
        report.append(f"  Dropout率: {model_info['dropout_rate']}")
        report.append(f"  总参数数: {model_info['total_parameters']:,}")
        report.append(f"  可训练参数数: {model_info['trainable_parameters']:,}")
        report.append("")
        
        # 性能指标
        datasets = [("训练集", train_metrics), ("验证集", val_metrics), ("测试集", test_metrics)]
        
        for dataset_name, metrics in datasets:
            report.append(f"{dataset_name}性能:")
            report.append(f"  MSE: {metrics['mse']:.6f}")
            report.append(f"  RMSE: {metrics['rmse']:.6f}")
            report.append(f"  MAE: {metrics['mae']:.6f}")
            report.append(f"  R²: {metrics['r2']:.6f}")
            report.append(f"  MAPE: {metrics['mape']:.2f}%")
            report.append(f"  SMAPE: {metrics['smape']:.2f}%")
            report.append(f"  最大误差: {metrics['max_error']:.6f}")
            report.append("")
        
        # 保存报告
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
            
        logger.info(f"评估报告已保存到: {save_path}")
        
        # 也打印到控制台
        print('\n'.join(report))