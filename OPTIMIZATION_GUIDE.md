# MLP数值预测模型优化指南

本文档提供了针对不同数据特性和需求的网络结构调整、超参数优化以及二次开发的详细指导。

## 📊 数据特性分析与网络结构调整

### 1. 根据数据规模调整网络结构

#### 小数据集 (< 1000 样本)
```yaml
# config.yaml 建议配置
model:
  hidden_layers: [32, 16]  # 较小的网络避免过拟合
  dropout_rate: 0.1        # 较低的dropout
  activation: "relu"

training:
  batch_size: 16           # 较小的batch size
  epochs: 200              # 更多的epochs
  learning_rate: 0.01      # 较高的学习率
  early_stopping_patience: 20
```

**调整原因**：
- 小网络减少参数数量，降低过拟合风险
- 小batch size增加梯度噪声，有助于泛化
- 更多epochs确保充分训练

#### 中等数据集 (1000-10000 样本)
```yaml
# 默认配置适用
model:
  hidden_layers: [128, 64, 32]
  dropout_rate: 0.2
  activation: "relu"

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
```

#### 大数据集 (> 10000 样本)
```yaml
model:
  hidden_layers: [256, 128, 64, 32]  # 更深的网络
  dropout_rate: 0.3                  # 更高的dropout
  activation: "relu"
  batch_norm: true                   # 启用批标准化

training:
  batch_size: 64                     # 更大的batch size
  epochs: 50                         # 较少的epochs
  learning_rate: 0.0001              # 较低的学习率
```

### 2. 根据特征维度调整

#### 高维特征 (> 100 维)
```python
# 在data_processor.py中添加特征选择
from sklearn.feature_selection import SelectKBest, f_regression

class DataProcessor:
    def feature_selection(self, k=50):
        """特征选择，保留最重要的k个特征"""
        selector = SelectKBest(score_func=f_regression, k=k)
        self.X_processed = selector.fit_transform(self.X_processed, self.y_processed)
        self.feature_selector = selector
        logger.info(f"特征选择完成，保留{k}个特征")
```

**网络结构建议**：
```yaml
model:
  hidden_layers: [512, 256, 128, 64]  # 第一层较大，逐渐减小
  dropout_rate: 0.4                   # 高dropout防止过拟合
```

#### 低维特征 (< 10 维)
```yaml
model:
  hidden_layers: [64, 32]             # 简单网络结构
  dropout_rate: 0.1                   # 低dropout
```

### 3. 根据目标变量特性调整

#### 多目标回归 (m > 5)
```python
# 在mlp_model.py中添加多任务学习支持
class MultiTaskMLPModel(MLPModel):
    def __init__(self, input_dim, output_dims_list, shared_layers=[128, 64], **kwargs):
        """
        多任务MLP模型
        
        Args:
            output_dims_list: 每个任务的输出维度列表
            shared_layers: 共享层结构
        """
        super().__init__(input_dim, sum(output_dims_list), shared_layers, **kwargs)
        
        # 为每个任务创建专门的输出层
        self.task_heads = nn.ModuleList()
        for output_dim in output_dims_list:
            self.task_heads.append(nn.Linear(shared_layers[-1], output_dim))
```

#### 目标值范围差异大
```python
# 在data_processor.py中添加目标值分别标准化
def normalize_targets_separately(self):
    """对每个目标变量分别进行标准化"""
    self.target_scalers = []
    normalized_targets = []
    
    for i in range(self.y_raw.shape[1]):
        scaler = StandardScaler()
        target_normalized = scaler.fit_transform(self.y_raw[:, i:i+1])
        normalized_targets.append(target_normalized)
        self.target_scalers.append(scaler)
    
    self.y_processed = np.hstack(normalized_targets)
```

## 🎯 超参数优化策略

### 1. 系统性超参数搜索

```python
# 创建hyperparameter_tuning.py
import optuna
from sklearn.model_selection import cross_val_score

class HyperparameterTuner:
    def __init__(self, base_config, X, y):
        self.base_config = base_config
        self.X = X
        self.y = y
    
    def objective(self, trial):
        """Optuna优化目标函数"""
        # 建议搜索空间
        config = self.base_config.copy()
        
        # 网络结构参数
        n_layers = trial.suggest_int('n_layers', 2, 5)
        hidden_sizes = []
        for i in range(n_layers):
            size = trial.suggest_int(f'hidden_size_{i}', 16, 512, log=True)
            hidden_sizes.append(size)
        
        config['model']['hidden_layers'] = hidden_sizes
        config['model']['dropout_rate'] = trial.suggest_float('dropout_rate', 0.0, 0.5)
        config['model']['activation'] = trial.suggest_categorical('activation', 
                                                                 ['relu', 'tanh', 'leaky_relu'])
        
        # 训练参数
        config['training']['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        config['training']['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        config['training']['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        
        # 交叉验证评估
        score = self.evaluate_config(config)
        return score
    
    def tune(self, n_trials=100):
        """执行超参数优化"""
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        return study.best_params, study.best_value
```

### 2. 学习率调度策略

```python
# 在trainer.py中添加更多学习率调度选项
def _create_advanced_scheduler(self):
    """创建高级学习率调度器"""
    scheduler_type = self.config.get('scheduler', {}).get('type', 'plateau')
    
    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config['training']['epochs']
        )
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=30,
            gamma=0.1
        )
    elif scheduler_type == 'exponential':
        return optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=0.95
        )
    else:
        return self._create_scheduler()  # 默认plateau
```

### 3. 早停策略优化

```python
# 在mlp_model.py中添加高级早停
class AdvancedEarlyStopping(EarlyStopping):
    def __init__(self, patience=10, min_delta=1e-6, warmup_epochs=10, **kwargs):
        super().__init__(patience, min_delta, **kwargs)
        self.warmup_epochs = warmup_epochs
        self.epoch_count = 0
    
    def __call__(self, val_loss, model):
        self.epoch_count += 1
        
        # 预热期不进行早停
        if self.epoch_count <= self.warmup_epochs:
            return False
            
        return super().__call__(val_loss, model)
```

## 🔧 二次开发指南

### 1. 模型架构扩展

#### 添加注意力机制
```python
# 在mlp_model.py中添加
class AttentionMLP(MLPModel):
    def __init__(self, *args, use_attention=True, **kwargs):
        super().__init__(*args, **kwargs)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.hidden_layers[0],
                num_heads=8,
                batch_first=True
            )
    
    def forward(self, x):
        # 标准MLP前向传播
        x = super().forward_to_last_hidden(x)  # 需要修改基类
        
        # 添加注意力机制
        if hasattr(self, 'attention'):
            x = x.unsqueeze(1)  # 添加序列维度
            x, _ = self.attention(x, x, x)
            x = x.squeeze(1)
        
        return self.output_layer(x)
```

#### 残差连接
```python
class ResidualMLP(MLPModel):
    def forward(self, x):
        residual = x
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # 批标准化
            if self.batch_norm and self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            # 激活函数
            x = self._get_activation(x)
            
            # 残差连接（维度匹配时）
            if i > 0 and x.shape == residual.shape:
                x = x + residual
            residual = x
            
            # Dropout
            x = self.dropouts[i](x)
        
        return self.output_layer(x)
```

### 2. 损失函数扩展

```python
# 创建custom_losses.py
import torch
import torch.nn as nn

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        error = torch.abs(pred - target)
        quadratic = torch.clamp(error, max=self.delta)
        linear = error - quadratic
        return torch.mean(0.5 * quadratic**2 + self.delta * linear)

class QuantileLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, pred, target):
        # pred shape: (batch_size, len(quantiles))
        losses = []
        for i, q in enumerate(self.quantiles):
            error = target - pred[:, i:i+1]
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss)
        return torch.mean(torch.cat(losses, dim=1))
```

### 3. 数据增强策略

```python
# 在data_processor.py中添加
class DataAugmenter:
    def __init__(self, noise_std=0.01, dropout_rate=0.1):
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate
    
    def add_noise(self, X):
        """添加高斯噪声"""
        noise = np.random.normal(0, self.noise_std, X.shape)
        return X + noise
    
    def feature_dropout(self, X):
        """随机丢弃特征"""
        mask = np.random.random(X.shape) > self.dropout_rate
        return X * mask
    
    def mixup(self, X, y, alpha=0.2):
        """Mixup数据增强"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = X.shape[0]
        index = np.random.permutation(batch_size)
        
        mixed_X = lam * X + (1 - lam) * X[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_X, mixed_y
```

### 4. 模型解释性工具

```python
# 创建model_interpretation.py
import shap
import numpy as np
from sklearn.inspection import permutation_importance

class ModelInterpreter:
    def __init__(self, model, X_background):
        self.model = model
        self.X_background = X_background
        
    def shap_analysis(self, X_test):
        """SHAP值分析"""
        # 创建SHAP解释器
        explainer = shap.DeepExplainer(self.model, self.X_background)
        shap_values = explainer.shap_values(X_test)
        
        return shap_values
    
    def feature_importance(self, X_test, y_test):
        """特征重要性分析"""
        def model_predict(X):
            with torch.no_grad():
                return self.model(torch.FloatTensor(X)).numpy()
        
        result = permutation_importance(
            model_predict, X_test, y_test,
            n_repeats=10, random_state=42
        )
        
        return result.importances_mean, result.importances_std
```

## 📈 性能优化建议

### 1. 计算优化

```python
# 在trainer.py中添加混合精度训练
from torch.cuda.amp import autocast, GradScaler

class OptimizedMLPTrainer(MLPTrainer):
    def __init__(self, *args, use_amp=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_amp = use_amp and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
    
    def train_epoch_optimized(self, train_loader):
        """优化的训练epoch"""
        self.model.train()
        total_loss = 0.0
        
        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
```

### 2. 内存优化

```python
# 梯度累积减少内存使用
def train_with_gradient_accumulation(self, train_loader, accumulation_steps=4):
    """梯度累积训练"""
    self.model.train()
    total_loss = 0.0
    
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(self.device), target.to(self.device)
        
        output = self.model(data)
        loss = self.criterion(output, target) / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
    
    return total_loss / len(train_loader)
```

## 🎨 可视化增强

```python
# 创建advanced_visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px

class AdvancedVisualizer:
    def __init__(self, save_dir="advanced_plots"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_feature_correlation_heatmap(self, X, feature_names=None):
        """特征相关性热力图"""
        corr_matrix = np.corrcoef(X.T)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, 
                   xticklabels=feature_names, 
                   yticklabels=feature_names,
                   annot=True, cmap='coolwarm', center=0)
        plt.title('特征相关性矩阵')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/feature_correlation.png", dpi=300)
        plt.show()
    
    def plot_learning_curves_interactive(self, history):
        """交互式学习曲线"""
        epochs = list(range(1, len(history['train_loss']) + 1))
        
        fig = go.Figure()
        
        # 训练损失
        fig.add_trace(go.Scatter(
            x=epochs, y=history['train_loss'],
            mode='lines', name='训练损失',
            line=dict(color='blue')
        ))
        
        # 验证损失
        fig.add_trace(go.Scatter(
            x=epochs, y=history['val_loss'],
            mode='lines', name='验证损失',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='模型训练历史',
            xaxis_title='Epoch',
            yaxis_title='损失',
            hovermode='x unified'
        )
        
        fig.write_html(f"{self.save_dir}/learning_curves.html")
        fig.show()
    
    def plot_prediction_distribution(self, y_true, y_pred):
        """预测分布对比"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 真实值分布
        axes[0].hist(y_true.flatten(), bins=50, alpha=0.7, label='真实值')
        axes[0].hist(y_pred.flatten(), bins=50, alpha=0.7, label='预测值')
        axes[0].set_xlabel('值')
        axes[0].set_ylabel('频次')
        axes[0].set_title('值分布对比')
        axes[0].legend()
        
        # Q-Q图
        from scipy import stats
        stats.probplot(y_true.flatten() - y_pred.flatten(), dist="norm", plot=axes[1])
        axes[1].set_title('残差Q-Q图')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/prediction_distribution.png", dpi=300)
        plt.show()
```

## 🔄 持续集成和部署

```python
# 创建model_deployment.py
import torch
import onnx
import onnxruntime as ort

class ModelDeployment:
    def __init__(self, model, example_input):
        self.model = model
        self.example_input = example_input
    
    def export_to_onnx(self, filepath="model.onnx"):
        """导出为ONNX格式"""
        self.model.eval()
        torch.onnx.export(
            self.model,
            self.example_input,
            filepath,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                         'output': {0: 'batch_size'}}
        )
        
        # 验证ONNX模型
        onnx_model = onnx.load(filepath)
        onnx.checker.check_model(onnx_model)
        
        return filepath
    
    def create_inference_session(self, onnx_path):
        """创建推理会话"""
        return ort.InferenceSession(onnx_path)
    
    def benchmark_inference(self, onnx_path, test_data, n_runs=100):
        """推理性能基准测试"""
        import time
        
        session = self.create_inference_session(onnx_path)
        
        # 预热
        for _ in range(10):
            session.run(None, {'input': test_data})
        
        # 基准测试
        start_time = time.time()
        for _ in range(n_runs):
            session.run(None, {'input': test_data})
        end_time = time.time()
        
        avg_time = (end_time - start_time) / n_runs
        return avg_time
```

## 📋 最佳实践总结

### 1. 数据预处理最佳实践
- **特征缩放**：始终对输入特征进行标准化
- **异常值处理**：使用IQR方法或Z-score检测异常值
- **特征工程**：考虑多项式特征、交互特征
- **数据验证**：实施数据质量检查

### 2. 模型设计最佳实践
- **网络深度**：从浅层开始，逐步增加深度
- **正则化**：结合Dropout、权重衰减、批标准化
- **激活函数**：ReLU系列通常是好的起点
- **初始化**：使用Xavier或He初始化

### 3. 训练最佳实践
- **学习率**：使用学习率查找器确定最优学习率
- **批大小**：平衡内存使用和梯度稳定性
- **早停**：防止过拟合，保存最佳模型
- **交叉验证**：确保模型泛化能力

### 4. 评估最佳实践
- **多指标评估**：不仅仅依赖单一指标
- **残差分析**：检查模型假设是否满足
- **特征重要性**：理解模型决策过程
- **模型解释**：使用SHAP等工具增强可解释性

这个优化指南为您的MLP数值预测项目提供了全面的改进方向和具体实现方案。根据您的具体需求和数据特性，可以选择性地应用这些优化策略。