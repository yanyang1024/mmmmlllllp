# 超参数调优指南

本文档提供了MLP数值预测模型的系统性超参数调优方法、工具和最佳实践。

## 🎯 超参数调优概述

### 关键超参数分类

#### 1. 网络架构参数
- **hidden_layers**: 隐藏层结构 `[128, 64, 32]`
- **activation**: 激活函数 `relu, tanh, sigmoid, leaky_relu`
- **dropout_rate**: Dropout比率 `0.0-0.5`
- **batch_norm**: 是否使用批标准化 `true/false`

#### 2. 训练参数
- **learning_rate**: 学习率 `1e-5 到 1e-1`
- **batch_size**: 批次大小 `16, 32, 64, 128, 256`
- **epochs**: 训练轮数 `50-500`
- **weight_decay**: 权重衰减 `1e-6 到 1e-2`

#### 3. 优化器参数
- **optimizer.type**: 优化器类型 `adam, sgd, rmsprop`
- **optimizer.momentum**: 动量 `0.9, 0.95, 0.99` (仅SGD)

#### 4. 调度器参数
- **scheduler.type**: 学习率调度器 `plateau, cosine, step`
- **early_stopping_patience**: 早停耐心值 `5-50`

## 🔍 数据特性驱动的调优策略

### 1. 基于数据规模的调优

```python
# 创建adaptive_tuning.py
import numpy as np
from typing import Dict, Any, Tuple

class AdaptiveTuner:
    """基于数据特性的自适应调优器"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.n_targets = y.shape[1] if y.ndim > 1 else 1
        
    def get_recommended_config(self) -> Dict[str, Any]:
        """根据数据特性推荐配置"""
        config = self._get_base_config()
        
        # 根据样本数量调整
        config = self._adjust_for_sample_size(config)
        
        # 根据特征维度调整
        config = self._adjust_for_feature_dimension(config)
        
        # 根据目标复杂度调整
        config = self._adjust_for_target_complexity(config)
        
        return config
    
    def _get_base_config(self) -> Dict[str, Any]:
        """基础配置"""
        return {
            'model': {
                'hidden_layers': [128, 64, 32],
                'activation': 'relu',
                'dropout_rate': 0.2,
                'batch_norm': False
            },
            'training': {
                'batch_size': 32,
                'epochs': 100,
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'early_stopping_patience': 10
            },
            'optimizer': {
                'type': 'adam'
            }
        }
    
    def _adjust_for_sample_size(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """根据样本数量调整配置"""
        if self.n_samples < 500:
            # 小数据集：简单模型，防止过拟合
            config['model']['hidden_layers'] = [64, 32]
            config['model']['dropout_rate'] = 0.1
            config['training']['batch_size'] = min(16, self.n_samples // 4)
            config['training']['learning_rate'] = 0.01
            config['training']['epochs'] = 200
            config['training']['early_stopping_patience'] = 20
            
        elif self.n_samples < 2000:
            # 中等数据集：标准配置
            config['training']['batch_size'] = min(32, self.n_samples // 8)
            
        else:
            # 大数据集：复杂模型，大批次
            config['model']['hidden_layers'] = [256, 128, 64, 32]
            config['model']['dropout_rate'] = 0.3
            config['model']['batch_norm'] = True
            config['training']['batch_size'] = min(128, self.n_samples // 16)
            config['training']['learning_rate'] = 0.0001
            config['training']['epochs'] = 50
        
        return config
    
    def _adjust_for_feature_dimension(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """根据特征维度调整配置"""
        if self.n_features > 100:
            # 高维特征：需要更大的第一层
            first_layer_size = min(512, self.n_features * 2)
            config['model']['hidden_layers'][0] = first_layer_size
            config['model']['dropout_rate'] = min(0.5, config['model']['dropout_rate'] + 0.1)
            
        elif self.n_features < 10:
            # 低维特征：简化网络
            config['model']['hidden_layers'] = [64, 32]
            config['model']['dropout_rate'] = max(0.1, config['model']['dropout_rate'] - 0.1)
        
        return config
    
    def _adjust_for_target_complexity(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """根据目标复杂度调整配置"""
        if self.n_targets > 5:
            # 多目标：增加网络容量
            config['model']['hidden_layers'] = [
                layer * 2 for layer in config['model']['hidden_layers']
            ]
            config['training']['learning_rate'] *= 0.5  # 降低学习率
            
        # 分析目标值的分布复杂度
        target_std = np.std(self.y, axis=0)
        if np.any(target_std > 10):  # 高方差目标
            config['training']['learning_rate'] *= 0.5
            config['training']['early_stopping_patience'] += 5
        
        return config
    
    def analyze_data_characteristics(self) -> Dict[str, Any]:
        """分析数据特性"""
        characteristics = {
            'sample_size': self.n_samples,
            'feature_dimension': self.n_features,
            'target_dimension': self.n_targets,
            'feature_correlation': self._analyze_feature_correlation(),
            'target_distribution': self._analyze_target_distribution(),
            'data_complexity': self._estimate_data_complexity()
        }
        
        return characteristics
    
    def _analyze_feature_correlation(self) -> Dict[str, float]:
        """分析特征相关性"""
        corr_matrix = np.corrcoef(self.X.T)
        
        # 移除对角线元素
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        correlations = corr_matrix[mask]
        
        return {
            'max_correlation': np.max(np.abs(correlations)),
            'mean_correlation': np.mean(np.abs(correlations)),
            'high_correlation_pairs': np.sum(np.abs(correlations) > 0.8)
        }
    
    def _analyze_target_distribution(self) -> Dict[str, Any]:
        """分析目标分布"""
        return {
            'mean': np.mean(self.y, axis=0).tolist(),
            'std': np.std(self.y, axis=0).tolist(),
            'skewness': self._calculate_skewness(self.y),
            'range': (np.min(self.y, axis=0).tolist(), np.max(self.y, axis=0).tolist())
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> list:
        """计算偏度"""
        from scipy import stats
        if data.ndim == 1:
            return [stats.skew(data)]
        else:
            return [stats.skew(data[:, i]) for i in range(data.shape[1])]
    
    def _estimate_data_complexity(self) -> str:
        """估计数据复杂度"""
        # 简单的复杂度估计
        feature_ratio = self.n_features / self.n_samples
        
        if feature_ratio > 0.5:
            return "high"  # 高维小样本
        elif feature_ratio > 0.1:
            return "medium"
        else:
            return "low"

# 使用示例
def get_adaptive_config(X, y):
    """获取自适应配置"""
    tuner = AdaptiveTuner(X, y)
    
    # 分析数据特性
    characteristics = tuner.analyze_data_characteristics()
    print("数据特性分析:")
    for key, value in characteristics.items():
        print(f"  {key}: {value}")
    
    # 获取推荐配置
    recommended_config = tuner.get_recommended_config()
    print("\n推荐配置:")
    print(yaml.dump(recommended_config, default_flow_style=False))
    
    return recommended_config
```

### 2. 自动超参数搜索

```python
# 创建hyperparameter_search.py
import optuna
import numpy as np
from sklearn.model_selection import KFold
from typing import Dict, Any, Callable
import yaml
import joblib

class HyperparameterSearcher:
    """超参数搜索器"""
    
    def __init__(self, 
                 X: np.ndarray, 
                 y: np.ndarray,
                 base_config: Dict[str, Any],
                 cv_folds: int = 5,
                 n_trials: int = 100):
        self.X = X
        self.y = y
        self.base_config = base_config
        self.cv_folds = cv_folds
        self.n_trials = n_trials
        self.study = None
        
    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """定义搜索空间"""
        config = self.base_config.copy()
        
        # 网络架构搜索
        n_layers = trial.suggest_int('n_layers', 2, 5)
        hidden_layers = []
        
        # 第一层大小
        first_layer = trial.suggest_int('first_layer', 32, 512, log=True)
        hidden_layers.append(first_layer)
        
        # 后续层递减
        current_size = first_layer
        for i in range(1, n_layers):
            # 确保层大小递减
            max_size = max(16, current_size // 2)
            layer_size = trial.suggest_int(f'layer_{i}', 16, max_size, log=True)
            hidden_layers.append(layer_size)
            current_size = layer_size
        
        config['model']['hidden_layers'] = hidden_layers
        
        # 其他超参数
        config['model']['activation'] = trial.suggest_categorical(
            'activation', ['relu', 'tanh', 'leaky_relu']
        )
        config['model']['dropout_rate'] = trial.suggest_float('dropout_rate', 0.0, 0.5)
        config['model']['batch_norm'] = trial.suggest_categorical('batch_norm', [True, False])
        
        # 训练参数
        config['training']['learning_rate'] = trial.suggest_float(
            'learning_rate', 1e-5, 1e-1, log=True
        )
        config['training']['batch_size'] = trial.suggest_categorical(
            'batch_size', [16, 32, 64, 128]
        )
        config['training']['weight_decay'] = trial.suggest_float(
            'weight_decay', 1e-6, 1e-2, log=True
        )
        
        # 优化器
        config['optimizer']['type'] = trial.suggest_categorical(
            'optimizer', ['adam', 'sgd', 'rmsprop']
        )
        
        if config['optimizer']['type'] == 'sgd':
            config['optimizer']['momentum'] = trial.suggest_float('momentum', 0.8, 0.99)
        
        return config
    
    def objective(self, trial: optuna.Trial) -> float:
        """优化目标函数"""
        config = self.define_search_space(trial)
        
        # 交叉验证
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X)):
            X_train_fold, X_val_fold = self.X[train_idx], self.X[val_idx]
            y_train_fold, y_val_fold = self.y[train_idx], self.y[val_idx]
            
            try:
                # 训练模型
                score = self._train_and_evaluate(
                    config, X_train_fold, y_train_fold, X_val_fold, y_val_fold
                )
                cv_scores.append(score)
                
            except Exception as e:
                # 如果训练失败，返回很差的分数
                print(f"Trial {trial.number} fold {fold} failed: {e}")
                return -1000.0
        
        # 返回平均CV分数
        mean_score = np.mean(cv_scores)
        
        # 报告中间结果
        trial.report(mean_score, fold)
        
        # 检查是否应该剪枝
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return mean_score
    
    def _train_and_evaluate(self, 
                           config: Dict[str, Any],
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_val: np.ndarray,
                           y_val: np.ndarray) -> float:
        """训练和评估模型"""
        from data_processor import DataProcessor
        from mlp_model import create_model_from_config
        from trainer import MLPTrainer
        
        # 创建数据处理器
        processor = DataProcessor(config)
        processor.load_data_from_arrays(
            np.vstack([X_train, X_val]), 
            np.vstack([y_train, y_val])
        )
        processor.normalize_data()
        
        # 手动分割数据（因为我们已经有了分割）
        train_size = len(X_train)
        X_processed = processor.X_processed
        y_processed = processor.y_processed
        
        X_train_proc = X_processed[:train_size]
        X_val_proc = X_processed[train_size:]
        y_train_proc = y_processed[:train_size]
        y_val_proc = y_processed[train_size:]
        
        # 创建数据加载器
        train_loader, val_loader, _ = processor.create_data_loaders(
            X_train_proc, X_val_proc, X_val_proc,  # 使用val作为test
            y_train_proc, y_val_proc, y_val_proc
        )
        
        # 创建模型
        model = create_model_from_config(config, processor.input_dim, processor.output_dim)
        
        # 创建训练器
        trainer = MLPTrainer(model, config)
        
        # 训练模型（较少的epochs用于搜索）
        config_copy = config.copy()
        config_copy['training']['epochs'] = min(50, config['training']['epochs'])
        config_copy['training']['early_stopping_patience'] = 5
        
        trainer.config = config_copy
        history = trainer.train(train_loader, val_loader)
        
        # 返回最佳验证R²分数
        best_r2 = max([metrics['r2'] for metrics in history['val_metrics']])
        return best_r2
    
    def search(self, 
               study_name: str = "mlp_hyperparameter_search",
               storage: str = None) -> Dict[str, Any]:
        """执行超参数搜索"""
        # 创建或加载study
        if storage:
            self.study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                load_if_exists=True,
                direction='maximize',
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            )
        else:
            self.study = optuna.create_study(
                direction='maximize',
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            )
        
        # 执行优化
        self.study.optimize(self.objective, n_trials=self.n_trials)
        
        # 返回最佳参数
        best_params = self.study.best_params
        best_config = self._params_to_config(best_params)
        
        return {
            'best_params': best_params,
            'best_config': best_config,
            'best_score': self.study.best_value,
            'study': self.study
        }
    
    def _params_to_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """将参数转换为配置格式"""
        config = self.base_config.copy()
        
        # 重建hidden_layers
        n_layers = params['n_layers']
        hidden_layers = [params['first_layer']]
        for i in range(1, n_layers):
            hidden_layers.append(params[f'layer_{i}'])
        
        config['model']['hidden_layers'] = hidden_layers
        config['model']['activation'] = params['activation']
        config['model']['dropout_rate'] = params['dropout_rate']
        config['model']['batch_norm'] = params['batch_norm']
        
        config['training']['learning_rate'] = params['learning_rate']
        config['training']['batch_size'] = params['batch_size']
        config['training']['weight_decay'] = params['weight_decay']
        
        config['optimizer']['type'] = params['optimizer']
        if params['optimizer'] == 'sgd':
            config['optimizer']['momentum'] = params.get('momentum', 0.9)
        
        return config
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """保存搜索结果"""
        # 保存最佳配置
        with open(f"{filepath}_best_config.yaml", 'w') as f:
            yaml.dump(results['best_config'], f, default_flow_style=False)
        
        # 保存study对象
        joblib.dump(results['study'], f"{filepath}_study.pkl")
        
        # 保存搜索摘要
        summary = {
            'best_score': results['best_score'],
            'best_params': results['best_params'],
            'n_trials': len(results['study'].trials),
            'study_name': results['study'].study_name
        }
        
        with open(f"{filepath}_summary.yaml", 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
    
    def plot_optimization_history(self, save_path: str = None):
        """绘制优化历史"""
        if self.study is None:
            raise ValueError("需要先执行搜索")
        
        import matplotlib.pyplot as plt
        
        # 优化历史
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 目标值历史
        trials = self.study.trials
        values = [trial.value for trial in trials if trial.value is not None]
        ax1.plot(values)
        ax1.set_xlabel('Trial')
        ax1.set_ylabel('Objective Value (R²)')
        ax1.set_title('Optimization History')
        ax1.grid(True)
        
        # 参数重要性
        try:
            importance = optuna.importance.get_param_importances(self.study)
            params = list(importance.keys())[:10]  # 前10个重要参数
            importances = [importance[param] for param in params]
            
            ax2.barh(params, importances)
            ax2.set_xlabel('Importance')
            ax2.set_title('Parameter Importance')
        except:
            ax2.text(0.5, 0.5, 'Parameter importance\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# 使用示例
def run_hyperparameter_search(X, y, base_config, n_trials=50):
    """运行超参数搜索"""
    searcher = HyperparameterSearcher(X, y, base_config, n_trials=n_trials)
    
    print(f"开始超参数搜索，共{n_trials}次试验...")
    results = searcher.search()
    
    print(f"\n搜索完成！")
    print(f"最佳分数: {results['best_score']:.4f}")
    print(f"最佳参数: {results['best_params']}")
    
    # 保存结果
    searcher.save_results(results, "hyperparameter_search_results")
    
    # 绘制优化历史
    searcher.plot_optimization_history("optimization_history.png")
    
    return results
```

### 3. 学习率查找器

```python
# 创建learning_rate_finder.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import math

class LearningRateFinder:
    """学习率查找器"""
    
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # 保存初始状态
        self.initial_state = model.state_dict().copy()
        self.initial_optimizer_state = optimizer.state_dict().copy()
    
    def find_lr(self, 
                train_loader,
                start_lr: float = 1e-7,
                end_lr: float = 10,
                num_iter: int = None,
                smooth_f: float = 0.05,
                diverge_th: int = 5) -> Tuple[List[float], List[float]]:
        """
        查找最优学习率
        
        Args:
            train_loader: 训练数据加载器
            start_lr: 起始学习率
            end_lr: 结束学习率
            num_iter: 迭代次数
            smooth_f: 平滑因子
            diverge_th: 发散阈值
            
        Returns:
            学习率列表和损失列表
        """
        if num_iter is None:
            num_iter = len(train_loader)
        
        # 计算学习率乘数
        mult = (end_lr / start_lr) ** (1 / num_iter)
        
        # 初始化
        lr = start_lr
        self.optimizer.param_groups[0]['lr'] = lr
        
        avg_loss = 0.0
        best_loss = 0.0
        batch_num = 0
        losses = []
        lrs = []
        
        # 训练循环
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= num_iter:
                break
                
            batch_num += 1
            
            # 前向传播
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            
            # 计算平滑损失
            if batch_idx == 0:
                avg_loss = loss.item()
                best_loss = avg_loss
            else:
                avg_loss = smooth_f * loss.item() + (1 - smooth_f) * avg_loss
            
            # 检查是否发散
            if batch_num > 1 and avg_loss > diverge_th * best_loss:
                print(f"学习率查找在第{batch_num}次迭代时停止，损失发散")
                break
            
            # 更新最佳损失
            if avg_loss < best_loss or batch_num == 1:
                best_loss = avg_loss
            
            # 记录
            losses.append(avg_loss)
            lrs.append(lr)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 更新学习率
            lr *= mult
            self.optimizer.param_groups[0]['lr'] = lr
        
        # 恢复初始状态
        self.model.load_state_dict(self.initial_state)
        self.optimizer.load_state_dict(self.initial_optimizer_state)
        
        return lrs, losses
    
    def plot(self, lrs: List[float], losses: List[float], save_path: str = None):
        """绘制学习率vs损失曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid(True)
        
        # 标记建议的学习率
        min_loss_idx = np.argmin(losses)
        suggested_lr = lrs[min_loss_idx] / 10  # 通常选择最小损失对应学习率的1/10
        
        plt.axvline(x=suggested_lr, color='red', linestyle='--', 
                   label=f'Suggested LR: {suggested_lr:.2e}')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return suggested_lr
    
    def suggest_lr(self, lrs: List[float], losses: List[float]) -> float:
        """建议最优学习率"""
        # 方法1: 最小损失对应学习率的1/10
        min_loss_idx = np.argmin(losses)
        lr_at_min_loss = lrs[min_loss_idx]
        
        # 方法2: 最大梯度下降速率对应的学习率
        gradients = np.gradient(losses)
        max_gradient_idx = np.argmin(gradients)  # 最负的梯度
        lr_at_max_gradient = lrs[max_gradient_idx]
        
        # 综合建议
        suggested_lr = min(lr_at_min_loss / 10, lr_at_max_gradient)
        
        print(f"最小损失对应学习率: {lr_at_min_loss:.2e}")
        print(f"最大梯度下降对应学习率: {lr_at_max_gradient:.2e}")
        print(f"建议学习率: {suggested_lr:.2e}")
        
        return suggested_lr

# 使用示例
def find_optimal_learning_rate(model, train_loader, config):
    """查找最优学习率"""
    from trainer import MLPTrainer
    
    # 创建临时训练器
    trainer = MLPTrainer(model, config)
    
    # 创建学习率查找器
    lr_finder = LearningRateFinder(
        model=trainer.model,
        optimizer=trainer.optimizer,
        criterion=trainer.criterion,
        device=trainer.device
    )
    
    # 查找学习率
    lrs, losses = lr_finder.find_lr(train_loader)
    
    # 绘制结果并获取建议
    suggested_lr = lr_finder.plot(lrs, losses, "learning_rate_finder.png")
    
    return suggested_lr
```

### 4. 批量大小优化

```python
# 创建batch_size_optimizer.py
import torch
import numpy as np
import time
from typing import Dict, List, Tuple

class BatchSizeOptimizer:
    """批量大小优化器"""
    
    def __init__(self, model, train_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def find_optimal_batch_size(self, 
                               batch_sizes: List[int] = None,
                               max_memory_usage: float = 0.8) -> Dict[str, any]:
        """
        查找最优批量大小
        
        Args:
            batch_sizes: 要测试的批量大小列表
            max_memory_usage: 最大内存使用率
            
        Returns:
            优化结果字典
        """
        if batch_sizes is None:
            batch_sizes = [8, 16, 32, 64, 128, 256, 512]
        
        results = []
        
        for batch_size in batch_sizes:
            print(f"测试批量大小: {batch_size}")
            
            try:
                result = self._test_batch_size(batch_size, max_memory_usage)
                if result:
                    results.append(result)
                    print(f"  成功: 时间={result['time_per_batch']:.3f}s, "
                          f"内存={result['memory_usage']:.1f}MB")
                else:
                    print(f"  失败: 内存不足或其他错误")
                    
            except Exception as e:
                print(f"  错误: {e}")
                continue
        
        if not results:
            raise RuntimeError("没有找到可用的批量大小")
        
        # 分析结果
        optimal_result = self._analyze_results(results)
        
        return optimal_result
    
    def _test_batch_size(self, batch_size: int, max_memory_usage: float) -> Dict:
        """测试特定批量大小"""
        from data_processor import DataProcessor
        from trainer import MLPTrainer
        
        # 创建新的数据加载器
        dataset = self.train_loader.dataset
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        # 创建训练器
        config_copy = self.config.copy()
        config_copy['training']['batch_size'] = batch_size
        trainer = MLPTrainer(self.model, config_copy)
        
        # 清空GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        
        # 测试几个批次
        times = []
        max_memory = 0
        
        trainer.model.train()
        
        for i, (data, target) in enumerate(test_loader):
            if i >= 5:  # 只测试5个批次
                break
                
            start_time = time.time()
            
            try:
                data, target = data.to(trainer.device), target.to(trainer.device)
                
                trainer.optimizer.zero_grad()
                output = trainer.model(data)
                loss = trainer.criterion(output, target)
                loss.backward()
                trainer.optimizer.step()
                
                end_time = time.time()
                times.append(end_time - start_time)
                
                # 检查内存使用
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated()
                    max_memory = max(max_memory, current_memory - initial_memory)
                    
                    # 检查是否超过内存限制
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    if current_memory / total_memory > max_memory_usage:
                        return None
                        
            except RuntimeError as e:
                if "out of memory" in str(e):
                    return None
                else:
                    raise e
        
        if not times:
            return None
        
        return {
            'batch_size': batch_size,
            'time_per_batch': np.mean(times),
            'time_std': np.std(times),
            'memory_usage': max_memory / (1024 * 1024),  # MB
            'throughput': batch_size / np.mean(times)  # samples/second
        }
    
    def _analyze_results(self, results: List[Dict]) -> Dict:
        """分析测试结果"""
        # 按吞吐量排序
        results_sorted = sorted(results, key=lambda x: x['throughput'], reverse=True)
        
        # 找到最佳批量大小（平衡吞吐量和内存使用）
        best_result = results_sorted[0]
        
        # 计算效率分数（吞吐量/内存使用）
        for result in results:
            result['efficiency_score'] = result['throughput'] / (result['memory_usage'] + 1)
        
        # 按效率分数排序
        efficiency_sorted = sorted(results, key=lambda x: x['efficiency_score'], reverse=True)
        most_efficient = efficiency_sorted[0]
        
        return {
            'all_results': results,
            'best_throughput': best_result,
            'most_efficient': most_efficient,
            'recommended': most_efficient,  # 推荐最高效的
            'summary': {
                'tested_batch_sizes': [r['batch_size'] for r in results],
                'best_throughput_batch_size': best_result['batch_size'],
                'most_efficient_batch_size': most_efficient['batch_size']
            }
        }
    
    def plot_results(self, results: Dict, save_path: str = None):
        """绘制批量大小优化结果"""
        import matplotlib.pyplot as plt
        
        all_results = results['all_results']
        batch_sizes = [r['batch_size'] for r in all_results]
        throughputs = [r['throughput'] for r in all_results]
        memory_usages = [r['memory_usage'] for r in all_results]
        efficiency_scores = [r['efficiency_score'] for r in all_results]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 吞吐量
        ax1.plot(batch_sizes, throughputs, 'bo-')
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Throughput (samples/sec)')
        ax1.set_title('Throughput vs Batch Size')
        ax1.grid(True)
        
        # 内存使用
        ax2.plot(batch_sizes, memory_usages, 'ro-')
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage vs Batch Size')
        ax2.grid(True)
        
        # 效率分数
        ax3.plot(batch_sizes, efficiency_scores, 'go-')
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Efficiency Score')
        ax3.set_title('Efficiency Score vs Batch Size')
        ax3.grid(True)
        
        # 推荐的批量大小
        recommended_bs = results['recommended']['batch_size']
        ax3.axvline(x=recommended_bs, color='red', linestyle='--', 
                   label=f'Recommended: {recommended_bs}')
        ax3.legend()
        
        # 时间对比
        times = [r['time_per_batch'] for r in all_results]
        ax4.plot(batch_sizes, times, 'mo-')
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Time per Batch (sec)')
        ax4.set_title('Time per Batch vs Batch Size')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# 使用示例
def optimize_batch_size(model, train_loader, config):
    """优化批量大小"""
    optimizer = BatchSizeOptimizer(model, train_loader, config)
    
    print("开始批量大小优化...")
    results = optimizer.find_optimal_batch_size()
    
    print(f"\n优化完成！")
    print(f"推荐批量大小: {results['recommended']['batch_size']}")
    print(f"吞吐量: {results['recommended']['throughput']:.1f} samples/sec")
    print(f"内存使用: {results['recommended']['memory_usage']:.1f} MB")
    
    # 绘制结果
    optimizer.plot_results(results, "batch_size_optimization.png")
    
    return results['recommended']['batch_size']
```

## 📊 超参数调优实战指南

### 1. 调优流程

```python
# 创建tuning_pipeline.py
def complete_hyperparameter_tuning(X, y, base_config, output_dir="tuning_results"):
    """完整的超参数调优流程"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== 开始完整超参数调优流程 ===\n")
    
    # 步骤1: 数据特性分析和自适应配置
    print("1. 数据特性分析...")
    tuner = AdaptiveTuner(X, y)
    characteristics = tuner.analyze_data_characteristics()
    adaptive_config = tuner.get_recommended_config()
    
    # 保存分析结果
    with open(f"{output_dir}/data_characteristics.yaml", 'w') as f:
        yaml.dump(characteristics, f, default_flow_style=False)
    
    with open(f"{output_dir}/adaptive_config.yaml", 'w') as f:
        yaml.dump(adaptive_config, f, default_flow_style=False)
    
    print(f"   数据特性已保存到 {output_dir}/data_characteristics.yaml")
    print(f"   自适应配置已保存到 {output_dir}/adaptive_config.yaml\n")
    
    # 步骤2: 学习率查找
    print("2. 学习率优化...")
    from data_processor import DataProcessor
    from mlp_model import create_model_from_config
    
    processor = DataProcessor(adaptive_config)
    processor.load_data_from_arrays(X, y)
    processor.normalize_data()
    
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data()
    train_loader, val_loader, test_loader = processor.create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    model = create_model_from_config(adaptive_config, processor.input_dim, processor.output_dim)
    optimal_lr = find_optimal_learning_rate(model, train_loader, adaptive_config)
    
    # 更新配置
    adaptive_config['training']['learning_rate'] = optimal_lr
    print(f"   最优学习率: {optimal_lr:.2e}\n")
    
    # 步骤3: 批量大小优化
    print("3. 批量大小优化...")
    optimal_batch_size = optimize_batch_size(model, train_loader, adaptive_config)
    adaptive_config['training']['batch_size'] = optimal_batch_size
    print(f"   最优批量大小: {optimal_batch_size}\n")
    
    # 步骤4: 全面超参数搜索
    print("4. 全面超参数搜索...")
    searcher = HyperparameterSearcher(X, y, adaptive_config, n_trials=100)
    search_results = searcher.search()
    
    # 保存搜索结果
    searcher.save_results(search_results, f"{output_dir}/hyperparameter_search")
    searcher.plot_optimization_history(f"{output_dir}/optimization_history.png")
    
    print(f"   最佳超参数搜索分数: {search_results['best_score']:.4f}")
    print(f"   搜索结果已保存到 {output_dir}/\n")
    
    # 步骤5: 最终验证
    print("5. 最终模型验证...")
    final_config = search_results['best_config']
    
    # 使用最佳配置训练最终模型
    final_model = create_model_from_config(final_config, processor.input_dim, processor.output_dim)
    from trainer import MLPTrainer
    final_trainer = MLPTrainer(final_model, final_config)
    
    # 完整训练
    final_config['training']['epochs'] = 100  # 恢复完整训练轮数
    final_trainer.config = final_config
    history = final_trainer.train(train_loader, val_loader)
    
    # 评估
    from evaluator import ModelEvaluator
    evaluator = ModelEvaluator(save_plots=True, plot_dir=output_dir)
    
    test_pred, test_true = final_trainer.predict(test_loader)
    test_pred_orig = processor.inverse_transform_predictions(test_pred)
    test_true_orig = processor.inverse_transform_predictions(test_true)
    
    final_metrics = evaluator.evaluate_model(test_true_orig, test_pred_orig, "final_test")
    
    # 保存最终结果
    final_results = {
        'final_config': final_config,
        'final_metrics': final_metrics,
        'model_info': final_model.get_model_info(),
        'training_summary': final_trainer.get_training_summary()
    }
    
    with open(f"{output_dir}/final_results.yaml", 'w') as f:
        yaml.dump(final_results, f, default_flow_style=False)
    
    # 保存最终模型
    final_trainer.save_model(f"{output_dir}/final_model.pth")
    processor.save_scalers(output_dir)
    
    print(f"   最终测试R²分数: {final_metrics['r2']:.4f}")
    print(f"   最终模型已保存到 {output_dir}/final_model.pth")
    
    print("\n=== 超参数调优完成 ===")
    
    return final_results

# 使用示例
if __name__ == "__main__":
    from data_processor import generate_sample_data
    
    # 生成示例数据
    X, y = generate_sample_data(n_samples=2000, n_features=15, n_targets=3)
    
    # 基础配置
    base_config = {
        'data': {'train_ratio': 0.8, 'val_ratio': 0.1, 'test_ratio': 0.1, 
                'random_seed': 42, 'normalize': True},
        'model': {'hidden_layers': [128, 64, 32], 'activation': 'relu', 
                 'dropout_rate': 0.2, 'batch_norm': False},
        'training': {'batch_size': 32, 'epochs': 100, 'learning_rate': 0.001, 
                    'weight_decay': 1e-5, 'early_stopping_patience': 10},
        'optimizer': {'type': 'adam'}
    }
    
    # 执行完整调优
    results = complete_hyperparameter_tuning(X, y, base_config)
```

这个超参数调优指南提供了系统性的调优方法和工具，帮助您根据数据特性找到最优的模型配置。通过自适应配置、学习率查找、批量大小优化和全面搜索的组合，可以显著提升模型性能。