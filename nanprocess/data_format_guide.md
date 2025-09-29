# 自定义数据集格式指南

本指南详细说明如何将您的数据转换为MLP项目可用的格式，包含多种数据源的处理方法和实际示例。

## 📋 数据格式要求

### 基本要求
- **输入数据 (X)**: `numpy.ndarray`, 形状为 `(n_samples, n_features)`, 数据类型 `float32`
- **输出数据 (y)**: `numpy.ndarray`, 形状为 `(n_samples, n_targets)`, 数据类型 `float32`
- **数值范围**: 建议进行标准化，项目会自动处理
- **缺失值**: 不允许存在 `NaN` 或 `inf` 值

### 数据维度说明
- `n_samples`: 样本数量 (建议 ≥ 100)
- `n_features`: 输入特征数量 (支持任意维度)
- `n_targets`: 输出目标数量 (支持单目标或多目标)

## 🔄 数据转换方法

### 方法1: 直接使用NumPy数组

```python
import numpy as np
from data_processor import DataProcessor

# 准备您的数据
X = your_input_features.astype(np.float32)   # 形状: (n_samples, n_features)
y = your_target_values.astype(np.float32)    # 形状: (n_samples, n_targets)

# 确保数据形状正确
if X.ndim == 1:
    X = X.reshape(-1, 1)
if y.ndim == 1:
    y = y.reshape(-1, 1)

# 使用数据处理器
config = load_config('config.yaml')
processor = DataProcessor(config)
processor.load_data_from_arrays(X, y)
```

### 方法2: 从Pandas DataFrame转换

```python
import pandas as pd
import numpy as np

# 从DataFrame提取数据
df = pd.read_csv('your_data.csv')

# 定义特征列和目标列
feature_columns = ['feature1', 'feature2', 'feature3']
target_columns = ['target1', 'target2']

# 提取数据
X = df[feature_columns].values.astype(np.float32)
y = df[target_columns].values.astype(np.float32)

# 处理缺失值 (如果有)
from sklearn.impute import SimpleImputer
if np.isnan(X).any():
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
```

### 方法3: 直接从CSV文件加载

```python
from data_processor import DataProcessor

# 使用内置CSV加载功能
processor = DataProcessor(config)
processor.load_data_from_csv(
    file_path='your_data.csv',
    target_columns=['target1', 'target2'],
    feature_columns=['feature1', 'feature2', 'feature3']  # 可选，默认使用所有其他列
)
```

## 📊 实际数据示例

### 示例1: 房价预测数据

```python
# 房价预测示例
import numpy as np

# 输入特征: 面积, 房间数, 楼层, 建造年份, 距离市中心
n_samples = 1000
area = np.random.normal(100, 30, n_samples)      # 面积 (平方米)
rooms = np.random.randint(1, 6, n_samples)       # 房间数
floor = np.random.randint(1, 21, n_samples)      # 楼层
year = np.random.randint(1990, 2024, n_samples)  # 建造年份
distance = np.random.exponential(5, n_samples)   # 距离市中心 (公里)

X = np.column_stack([area, rooms, floor, year, distance]).astype(np.float32)

# 输出目标: 房价, 租金
price = (area * 0.8 + rooms * 5 + (2024 - year) * 0.1 - distance * 2 + 
         np.random.normal(0, 5, n_samples))
rent = price * 50 + np.random.normal(0, 200, n_samples)

y = np.column_stack([price, rent]).astype(np.float32)

print(f"数据形状: X={X.shape}, y={y.shape}")
# 输出: 数据形状: X=(1000, 5), y=(1000, 2)
```

### 示例2: 时间序列数据转换

```python
# 将时间序列转换为监督学习问题
def create_sequences(data, lookback, forecast):
    """
    将时间序列数据转换为监督学习格式
    
    Args:
        data: 时间序列数据 (n_timesteps, n_features)
        lookback: 回看窗口大小
        forecast: 预测窗口大小
    
    Returns:
        X: (n_samples, lookback * n_features)
        y: (n_samples, forecast * n_targets)
    """
    X, y = [], []
    
    for i in range(lookback, len(data) - forecast):
        # 输入: 过去lookback个时间步的所有特征
        x_seq = data[i-lookback:i].flatten()
        
        # 输出: 未来forecast个时间步的目标特征
        y_seq = data[i:i+forecast, :2].flatten()  # 假设前2个特征是目标
        
        X.append(x_seq)
        y.append(y_seq)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# 使用示例
time_series_data = np.random.randn(1000, 4)  # 1000个时间步，4个特征
X, y = create_sequences(time_series_data, lookback=24, forecast=6)
print(f"转换后形状: X={X.shape}, y={y.shape}")
# 输出: 转换后形状: X=(970, 96), y=(970, 12)
```

### 示例3: 图像特征数据

```python
# 从预训练模型提取的图像特征
def prepare_image_features(feature_vectors, labels):
    """
    准备图像特征数据
    
    Args:
        feature_vectors: 图像特征向量列表或数组
        labels: 对应的标签或评分
    
    Returns:
        X, y: 格式化后的数据
    """
    X = np.array(feature_vectors, dtype=np.float32)
    
    # 如果标签是分类，转换为回归目标
    if isinstance(labels[0], str):
        # 示例: 将类别转换为数值
        label_map = {'low': 1, 'medium': 5, 'high': 9}
        y = np.array([label_map[label] for label in labels], dtype=np.float32)
        y = y.reshape(-1, 1)
    else:
        y = np.array(labels, dtype=np.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
    
    return X, y

# 使用示例
features = np.random.randn(500, 512)  # 500张图片，512维特征
scores = np.random.uniform(1, 10, 500)  # 质量评分 1-10
X, y = prepare_image_features(features, scores)
```

## 🔧 数据预处理建议

### 1. 数据清洗

```python
def clean_data(X, y):
    """清洗数据，移除异常值和缺失值"""
    # 检查缺失值
    if np.isnan(X).any() or np.isnan(y).any():
        print("警告: 发现缺失值")
        # 移除包含缺失值的样本
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
        X, y = X[mask], y[mask]
    
    # 检查无穷值
    if np.isinf(X).any() or np.isinf(y).any():
        print("警告: 发现无穷值")
        mask = ~(np.isinf(X).any(axis=1) | np.isinf(y).any(axis=1))
        X, y = X[mask], y[mask]
    
    # 移除异常值 (使用IQR方法)
    from scipy import stats
    z_scores = np.abs(stats.zscore(X, axis=0))
    mask = (z_scores < 3).all(axis=1)  # 保留z-score < 3的样本
    X, y = X[mask], y[mask]
    
    print(f"清洗后数据形状: X={X.shape}, y={y.shape}")
    return X, y
```

### 2. 特征工程

```python
def feature_engineering(X):
    """特征工程示例"""
    from sklearn.preprocessing import PolynomialFeatures
    
    # 添加多项式特征
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # 添加统计特征
    X_stats = np.column_stack([
        X.mean(axis=1),  # 每个样本的均值
        X.std(axis=1),   # 每个样本的标准差
        X.max(axis=1),   # 每个样本的最大值
        X.min(axis=1)    # 每个样本的最小值
    ])
    
    # 合并特征
    X_enhanced = np.column_stack([X, X_stats])
    
    return X_enhanced.astype(np.float32)
```

### 3. 数据验证

```python
def validate_data(X, y):
    """验证数据格式和质量"""
    checks = []
    
    # 检查数据类型
    checks.append(("X数据类型", X.dtype == np.float32))
    checks.append(("y数据类型", y.dtype == np.float32))
    
    # 检查数据形状
    checks.append(("X是2D数组", X.ndim == 2))
    checks.append(("y是2D数组", y.ndim == 2))
    checks.append(("样本数量匹配", X.shape[0] == y.shape[0]))
    
    # 检查数据质量
    checks.append(("X无缺失值", not np.isnan(X).any()))
    checks.append(("y无缺失值", not np.isnan(y).any()))
    checks.append(("X无无穷值", not np.isinf(X).any()))
    checks.append(("y无无穷值", not np.isinf(y).any()))
    
    # 检查样本数量
    checks.append(("样本数量充足", X.shape[0] >= 100))
    
    print("数据验证结果:")
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
    
    all_passed = all(result for _, result in checks)
    
    if all_passed:
        print("🎉 数据验证通过！可以开始训练。")
    else:
        print("⚠️  数据验证失败，请检查并修复问题。")
    
    return all_passed
```

## 📝 完整使用示例

```python
# 完整的数据准备和训练流程
import numpy as np
import yaml
from data_processor import DataProcessor
from mlp_model import create_model_from_config
from trainer import MLPTrainer

def train_with_your_data():
    """使用您的数据训练模型的完整流程"""
    
    # 步骤1: 准备数据
    # 替换为您的实际数据加载代码
    X = np.random.randn(1000, 10).astype(np.float32)  # 您的输入特征
    y = np.random.randn(1000, 3).astype(np.float32)   # 您的目标值
    
    # 步骤2: 数据验证
    if not validate_data(X, y):
        return False
    
    # 步骤3: 数据清洗 (可选)
    X, y = clean_data(X, y)
    
    # 步骤4: 特征工程 (可选)
    # X = feature_engineering(X)
    
    # 步骤5: 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 步骤6: 创建数据处理器
    processor = DataProcessor(config)
    processor.load_data_from_arrays(X, y)
    processor.normalize_data()
    
    # 步骤7: 分割数据
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data()
    train_loader, val_loader, test_loader = processor.create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # 步骤8: 创建和训练模型
    model = create_model_from_config(config, processor.input_dim, processor.output_dim)
    trainer = MLPTrainer(model, config)
    
    print("开始训练...")
    history = trainer.train(train_loader, val_loader)
    
    # 步骤9: 评估模型
    test_pred, test_true = trainer.predict(test_loader)
    test_pred_orig = processor.inverse_transform_predictions(test_pred)
    test_true_orig = processor.inverse_transform_predictions(test_true)
    
    from evaluator import ModelEvaluator
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(test_true_orig, test_pred_orig)
    
    print(f"训练完成！最终R²分数: {metrics['r2']:.4f}")
    
    return True

# 运行训练
if __name__ == "__main__":
    train_with_your_data()
```

## 🚨 NaN值处理指南

### NaN值的识别和分析

如果您的数据中存在NaN值，需要在训练前进行处理。项目提供了专门的NaN处理工具：

```python
from nan_handling_guide import NaNHandler

# 创建NaN处理器
handler = NaNHandler()

# 分析NaN分布模式
analysis = handler.analyze_nan_pattern(X, y, feature_names=['特征1', '特征2', ...])
handler.print_nan_summary(analysis)

# 可视化NaN模式
handler.visualize_nan_pattern(X, save_path="nan_analysis.png")
```

### NaN处理策略

#### 策略1: 删除含NaN的样本
```python
# 适用于: 完整样本比例较高(>70%)的情况
X_clean, y_clean = handler.strategy_1_remove_samples(X, y, threshold=0.5)
```

#### 策略2: 删除高缺失率特征
```python
# 适用于: 某些特征缺失率过高(>70%)的情况
X_clean, y_clean, remaining_features = handler.strategy_2_remove_features(
    X, y, threshold=0.7, feature_names=feature_names
)
```

#### 策略3: 简单插值填充
```python
# 适用于: 缺失值较少且随机分布的情况
X_clean, y_clean = handler.strategy_3_simple_imputation(X, y, strategy='mean')
# strategy可选: 'mean', 'median', 'most_frequent'
```

#### 策略4: KNN插值填充
```python
# 适用于: 特征间有相关性，缺失值适中的情况
X_clean, y_clean = handler.strategy_4_knn_imputation(X, y, n_neighbors=5)
```

#### 策略5: 迭代插值填充(MICE)
```python
# 适用于: 复杂缺失模式，样本量充足的情况
X_clean, y_clean = handler.strategy_5_iterative_imputation(X, y, max_iter=10)
```

#### 策略6: 混合方法(推荐)
```python
# 综合多种方法，适用于大多数情况
X_clean, y_clean = handler.strategy_6_hybrid_approach(X, y)
```

### 自动策略推荐

```python
# 获取基于数据特性的策略推荐
recommended_strategy = handler.recommend_strategy(analysis)
print(f"推荐策略: {recommended_strategy}")

# 比较不同策略的效果
comparison = handler.compare_strategies(X, y)
for strategy, result in comparison.items():
    print(f"{strategy}: 保留样本{result['samples_retained']*100:.1f}%")
```

### 完整的NaN处理流程

```python
def handle_nan_data(X, y):
    """处理含NaN数据的完整流程"""
    
    # 1. 分析NaN模式
    handler = NaNHandler()
    analysis = handler.analyze_nan_pattern(X, y)
    handler.print_nan_summary(analysis)
    
    # 2. 获取推荐策略
    strategy = handler.recommend_strategy(analysis)
    
    # 3. 应用处理策略
    if strategy == 'hybrid':
        X_clean, y_clean = handler.strategy_6_hybrid_approach(X, y)
    elif strategy == 'knn':
        X_clean, y_clean = handler.strategy_4_knn_imputation(X, y)
    # ... 其他策略
    
    # 4. 验证处理结果
    assert not np.isnan(X_clean).any(), "仍有NaN值"
    assert not np.isnan(y_clean).any(), "目标值仍有NaN"
    
    print(f"NaN处理完成: {X.shape} -> {X_clean.shape}")
    return X_clean, y_clean

# 使用示例
X_clean, y_clean = handle_nan_data(your_X_with_nan, your_y_with_nan)

# 继续正常的训练流程
processor = DataProcessor(config)
processor.load_data_from_arrays(X_clean, y_clean)
```

### NaN处理最佳实践

1. **先分析后处理**: 了解NaN的分布模式再选择策略
2. **保留足够数据**: 避免过度删除导致数据不足
3. **验证处理效果**: 确保处理后数据质量良好
4. **记录处理过程**: 便于后续数据预处理的一致性
5. **考虑业务含义**: NaN可能有特殊含义，不一定要填充

## ⚠️ 常见问题和解决方案

### 问题1: 数据类型错误
```
TypeError: can't convert np.ndarray of type numpy.object_
```
**解决方案**: 确保数据类型为数值型
```python
X = X.astype(np.float32)
y = y.astype(np.float32)
```

### 问题2: 数据维度错误
```
ValueError: Expected 2D array, got 1D array
```
**解决方案**: 调整数据维度
```python
if X.ndim == 1:
    X = X.reshape(-1, 1)
if y.ndim == 1:
    y = y.reshape(-1, 1)
```

### 问题3: 样本数量不匹配
```
ValueError: X and y must have the same number of samples
```
**解决方案**: 检查并对齐样本数量
```python
min_samples = min(len(X), len(y))
X = X[:min_samples]
y = y[:min_samples]
```

### 问题4: 内存不足
```
RuntimeError: CUDA out of memory
```
**解决方案**: 减小批次大小或数据量
```python
config['training']['batch_size'] = 16  # 减小批次大小
# 或者使用数据子集
X = X[:5000]  # 只使用前5000个样本
y = y[:5000]
```

## 🎯 最佳实践建议

1. **数据质量**: 确保数据无缺失值、异常值和重复样本
2. **特征缩放**: 让项目自动处理标准化，或手动进行特征缩放
3. **数据分布**: 检查目标变量的分布，考虑是否需要变换
4. **样本平衡**: 确保训练集中各类样本分布合理
5. **验证集**: 保留足够的验证数据用于模型选择
6. **文档记录**: 记录数据来源、预处理步骤和特征含义

通过遵循这个指南，您可以轻松地将任何格式的数据转换为项目可用的格式，并获得良好的训练效果！