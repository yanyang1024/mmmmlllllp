# MLP数值预测模型

一个基于PyTorch的多层感知机(MLP)数值预测系统，专为处理n个浮点数输入预测m个浮点数输出的回归任务而设计。

## 项目特点

- 🚀 **高性能**: 基于PyTorch实现，支持GPU加速
- 🔧 **高度可配置**: 通过YAML配置文件轻松调整模型参数
- 📊 **全面评估**: 提供多种评估指标和可视化功能
- 🛡️ **稳定训练**: 集成早停、学习率调度、梯度裁剪等技术
- 📈 **详细监控**: 完整的训练历史记录和日志系统
- 🔄 **易于使用**: 提供简洁的API和丰富的使用示例

## 项目结构

```
mlp/
├── config.yaml           # 配置文件
├── requirements.txt       # 依赖包列表
├── main.py               # 主程序入口
├── example_usage.py      # 使用示例
├── data_processor.py     # 数据处理模块
├── mlp_model.py          # MLP模型定义
├── trainer.py            # 模型训练器
├── evaluator.py          # 模型评估器
├── README.md             # 项目说明
├── models/               # 模型保存目录
├── logs/                 # 日志文件目录
└── plots/                # 图表保存目录
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 基本使用

```python
import numpy as np
from main import load_config, prepare_data, train_model, evaluate_model

# 加载配置
config = load_config('config.yaml')

# 准备数据（使用示例数据）
processor, data_loaders = prepare_data(config, data_source="sample")

# 训练模型
trainer = train_model(config, processor, data_loaders)

# 评估模型
results = evaluate_model(trainer, processor, data_loaders)
```

### 2. 使用自定义数据

```python
import numpy as np
from data_processor import DataProcessor
from mlp_model import create_model_from_config
from trainer import MLPTrainer

# 准备你的数据
X = np.random.randn(1000, 10).astype(np.float32)  # 1000个样本，10个特征
y = np.random.randn(1000, 3).astype(np.float32)   # 1000个样本，3个目标

# 数据处理
processor = DataProcessor(config)
processor.load_data_from_arrays(X, y)
processor.normalize_data()

# 分割数据
X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data()
train_loader, val_loader, test_loader = processor.create_data_loaders(
    X_train, X_val, X_test, y_train, y_val, y_test
)

# 创建和训练模型
model = create_model_from_config(config, processor.input_dim, processor.output_dim)
trainer = MLPTrainer(model, config)
history = trainer.train(train_loader, val_loader)
```

### 3. 命令行使用

```bash
# 训练模型
python main.py --mode train --config config.yaml --save_dir models

# 使用训练好的模型进行预测
python main.py --mode predict --model_path models/final_model.pth --config config.yaml
```

### 4. 运行示例

```bash
python example_usage.py
```

## 配置说明

主要配置参数说明：

```yaml
# 数据配置
data:
  train_ratio: 0.8        # 训练集比例
  val_ratio: 0.1          # 验证集比例
  test_ratio: 0.1         # 测试集比例
  normalize: true         # 是否标准化数据

# 模型配置
model:
  hidden_layers: [128, 64, 32]  # 隐藏层神经元数量
  dropout_rate: 0.2             # Dropout比率
  activation: "relu"            # 激活函数

# 训练配置
training:
  batch_size: 32               # 批次大小
  epochs: 100                  # 训练轮数
  learning_rate: 0.001         # 学习率
  early_stopping_patience: 10  # 早停耐心值
```

## 模型特性

### 支持的激活函数
- ReLU
- Tanh
- Sigmoid
- Leaky ReLU

### 支持的优化器
- Adam
- SGD
- RMSprop

### 支持的损失函数
- MSE (均方误差)
- MAE (平均绝对误差)
- Huber Loss

### 评估指标
- MSE (均方误差)
- RMSE (均方根误差)
- MAE (平均绝对误差)
- R² (决定系数)
- MAPE (平均绝对百分比误差)
- SMAPE (对称平均绝对百分比误差)
- 皮尔逊相关系数

## API参考

### DataProcessor类

```python
# 创建数据处理器
processor = DataProcessor(config)

# 从数组加载数据
processor.load_data_from_arrays(X, y)

# 从CSV文件加载数据
processor.load_data_from_csv('data.csv', target_columns=['target1', 'target2'])

# 数据标准化
processor.normalize_data()

# 数据分割
X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data()
```

### MLPModel类

```python
# 创建模型
model = MLPModel(
    input_dim=10,
    output_dim=3,
    hidden_layers=[128, 64, 32],
    activation='relu',
    dropout_rate=0.2
)

# 获取模型信息
info = model.get_model_info()
```

### MLPTrainer类

```python
# 创建训练器
trainer = MLPTrainer(model, config)

# 训练模型
history = trainer.train(train_loader, val_loader)

# 进行预测
predictions, targets = trainer.predict(test_loader)

# 单样本预测
pred = trainer.predict_single(x_new)
```

### ModelEvaluator类

```python
# 创建评估器
evaluator = ModelEvaluator(save_plots=True)

# 评估模型
metrics = evaluator.evaluate_model(y_true, y_pred, "test")

# 绘制图表
evaluator.plot_predictions_vs_actual(y_true, y_pred)
evaluator.plot_residuals(y_true, y_pred)
evaluator.plot_training_history(history)
```

## 高级功能

### 早停机制
自动监控验证损失，在性能不再提升时停止训练：

```python
early_stopping = EarlyStopping(
    patience=10,
    min_delta=1e-6,
    restore_best_weights=True
)
```

### 模型检查点
自动保存最佳模型：

```python
checkpoint = ModelCheckpoint(
    filepath='best_model.pth',
    monitor='val_loss',
    save_best_only=True
)
```

### 学习率调度
自动调整学习率：

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5
)
```

## 注意事项

1. **离线环境**: 项目设计为在离线环境中运行，不依赖在线服务
2. **数据格式**: 输入数据应为numpy数组，dtype为float32
3. **内存使用**: 大数据集建议调整batch_size以控制内存使用
4. **GPU支持**: 自动检测并使用可用的GPU进行训练

## 故障排除

### 常见问题

1. **内存不足**: 减小batch_size或hidden_layers大小
2. **训练不收敛**: 调整学习率或增加训练轮数
3. **过拟合**: 增加dropout_rate或减少模型复杂度
4. **欠拟合**: 增加模型复杂度或减少正则化

### 性能优化建议

1. 使用GPU加速训练
2. 合理设置batch_size（通常32-128）
3. 使用数据标准化
4. 启用早停机制避免过拟合
5. 使用学习率调度优化收敛

## 许可证

本项目采用MIT许可证。

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 更新日志

### v1.0.0
- 初始版本发布
- 支持基本的MLP训练和预测功能
- 完整的数据处理和评估系统
- 丰富的配置选项和使用示例