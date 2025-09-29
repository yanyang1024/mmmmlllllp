"""
自定义数据集示例
演示如何将不同格式的数据转换为MLP项目可用的格式
"""

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from typing import Tuple, List, Dict, Any

from data_processor import DataProcessor
from mlp_model import create_model_from_config
from trainer import MLPTrainer
from evaluator import ModelEvaluator


def example_1_numpy_arrays():
    """
    示例1: 从NumPy数组创建数据集
    适用场景: 已有处理好的数值数据
    """
    print("\n" + "="*60)
    print("示例1: 使用NumPy数组数据")
    print("="*60)
    
    # 模拟房价预测数据
    # 输入特征: 面积, 房间数, 楼层, 建造年份, 距离市中心距离
    np.random.seed(42)
    n_samples = 1000
    
    # 生成输入特征 (n_samples, 5)
    area = np.random.normal(100, 30, n_samples)  # 面积 (平方米)
    rooms = np.random.randint(1, 6, n_samples)   # 房间数
    floor = np.random.randint(1, 21, n_samples)  # 楼层
    year = np.random.randint(1990, 2024, n_samples)  # 建造年份
    distance = np.random.exponential(5, n_samples)   # 距离市中心 (公里)
    
    X = np.column_stack([area, rooms, floor, year, distance]).astype(np.float32)
    
    # 生成目标值 (n_samples, 2)
    # 目标1: 房价 (万元), 目标2: 租金 (元/月)
    price = (area * 0.8 + rooms * 5 + (2024 - year) * 0.1 - distance * 2 + 
             np.random.normal(0, 5, n_samples))
    rent = price * 50 + np.random.normal(0, 200, n_samples)
    
    y = np.column_stack([price, rent]).astype(np.float32)
    
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"输入特征: 面积, 房间数, 楼层, 建造年份, 距离市中心")
    print(f"输出目标: 房价(万元), 租金(元/月)")
    print(f"数据样例:")
    print(f"  X[0]: {X[0]}")
    print(f"  y[0]: {y[0]}")
    
    # 使用数据训练模型
    config = load_config()
    config['training']['epochs'] = 20  # 快速演示
    
    success = train_with_custom_data(X, y, config, "房价预测模型")
    
    return X, y, success


def example_2_pandas_dataframe():
    """
    示例2: 从Pandas DataFrame创建数据集
    适用场景: 结构化数据，需要特征选择和预处理
    """
    print("\n" + "="*60)
    print("示例2: 使用Pandas DataFrame数据")
    print("="*60)
    
    # 创建模拟的学生成绩预测数据
    np.random.seed(42)
    n_samples = 800
    
    # 创建DataFrame
    data = {
        'study_hours': np.random.normal(6, 2, n_samples),      # 学习时间
        'sleep_hours': np.random.normal(7, 1, n_samples),      # 睡眠时间
        'exercise_hours': np.random.normal(1, 0.5, n_samples), # 运动时间
        'family_income': np.random.normal(50000, 15000, n_samples), # 家庭收入
        'previous_score': np.random.normal(75, 10, n_samples), # 之前成绩
        'attendance_rate': np.random.uniform(0.7, 1.0, n_samples), # 出勤率
        'age': np.random.randint(16, 20, n_samples),           # 年龄
        'gender': np.random.choice([0, 1], n_samples),         # 性别 (0:女, 1:男)
        
        # 目标变量
        'math_score': 0,    # 数学成绩
        'english_score': 0, # 英语成绩
        'science_score': 0  # 科学成绩
    }
    
    df = pd.DataFrame(data)
    
    # 生成目标变量 (基于输入特征的复杂关系)
    df['math_score'] = (
        df['study_hours'] * 3 + 
        df['previous_score'] * 0.6 + 
        df['attendance_rate'] * 20 + 
        np.random.normal(0, 5, n_samples)
    ).clip(0, 100)
    
    df['english_score'] = (
        df['study_hours'] * 2.5 + 
        df['previous_score'] * 0.7 + 
        df['sleep_hours'] * 2 + 
        np.random.normal(0, 4, n_samples)
    ).clip(0, 100)
    
    df['science_score'] = (
        df['study_hours'] * 3.5 + 
        df['previous_score'] * 0.5 + 
        df['exercise_hours'] * 3 + 
        np.random.normal(0, 6, n_samples)
    ).clip(0, 100)
    
    print(f"DataFrame形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print(f"数据样例:")
    print(df.head(3))
    
    # 提取特征和目标
    feature_columns = ['study_hours', 'sleep_hours', 'exercise_hours', 
                      'family_income', 'previous_score', 'attendance_rate', 
                      'age', 'gender']
    target_columns = ['math_score', 'english_score', 'science_score']
    
    X = df[feature_columns].values.astype(np.float32)
    y = df[target_columns].values.astype(np.float32)
    
    print(f"\n提取后数据形状: X={X.shape}, y={y.shape}")
    print(f"特征列: {feature_columns}")
    print(f"目标列: {target_columns}")
    
    # 使用数据训练模型
    config = load_config()
    config['training']['epochs'] = 20
    
    success = train_with_custom_data(X, y, config, "学生成绩预测模型")
    
    return X, y, success


def example_3_csv_file():
    """
    示例3: 从CSV文件加载数据
    适用场景: 外部数据文件
    """
    print("\n" + "="*60)
    print("示例3: 从CSV文件加载数据")
    print("="*60)
    
    # 创建示例CSV文件
    csv_filename = "sample_data.csv"
    
    # 生成股票价格预测数据
    np.random.seed(42)
    n_samples = 600
    
    # 技术指标作为输入特征
    data = {
        'open_price': np.random.uniform(50, 150, n_samples),
        'high_price': np.random.uniform(55, 155, n_samples),
        'low_price': np.random.uniform(45, 145, n_samples),
        'volume': np.random.uniform(1000000, 10000000, n_samples),
        'rsi': np.random.uniform(20, 80, n_samples),          # RSI指标
        'macd': np.random.uniform(-2, 2, n_samples),          # MACD指标
        'bollinger_upper': np.random.uniform(60, 160, n_samples),
        'bollinger_lower': np.random.uniform(40, 140, n_samples),
        'moving_avg_5': np.random.uniform(52, 152, n_samples),
        'moving_avg_20': np.random.uniform(51, 151, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # 生成目标变量
    # 目标1: 下一日收盘价, 目标2: 价格变化百分比
    df['next_close'] = (
        df['open_price'] * 0.7 + 
        df['moving_avg_5'] * 0.2 + 
        df['rsi'] * 0.3 + 
        np.random.normal(0, 3, n_samples)
    )
    
    df['price_change_pct'] = (
        (df['next_close'] - df['open_price']) / df['open_price'] * 100
    )
    
    # 保存到CSV
    df.to_csv(csv_filename, index=False)
    print(f"已创建示例CSV文件: {csv_filename}")
    print(f"文件包含 {len(df)} 行数据，{len(df.columns)} 列")
    print(f"列名: {list(df.columns)}")
    
    # 从CSV加载数据
    feature_columns = ['open_price', 'high_price', 'low_price', 'volume', 
                      'rsi', 'macd', 'bollinger_upper', 'bollinger_lower',
                      'moving_avg_5', 'moving_avg_20']
    target_columns = ['next_close', 'price_change_pct']
    
    # 使用DataProcessor的CSV加载功能
    config = load_config()
    config['training']['epochs'] = 20
    
    processor = DataProcessor(config)
    processor.load_data_from_csv(csv_filename, target_columns, feature_columns)
    
    print(f"\n从CSV加载的数据形状: X={processor.X_raw.shape}, y={processor.y_raw.shape}")
    print(f"输入维度: {processor.input_dim}, 输出维度: {processor.output_dim}")
    
    # 训练模型
    success = train_with_processor(processor, config, "股票价格预测模型")
    
    return processor.X_raw, processor.y_raw, success


def example_4_time_series_data():
    """
    示例4: 时间序列数据转换
    适用场景: 时间序列预测任务
    """
    print("\n" + "="*60)
    print("示例4: 时间序列数据转换")
    print("="*60)
    
    # 生成时间序列数据 (如传感器数据)
    np.random.seed(42)
    n_timesteps = 1000
    
    # 生成多个传感器的时间序列
    time = np.arange(n_timesteps)
    
    # 传感器1: 温度 (有季节性)
    temp = 20 + 10 * np.sin(2 * np.pi * time / 365) + np.random.normal(0, 2, n_timesteps)
    
    # 传感器2: 湿度 (与温度相关)
    humidity = 60 - 0.5 * temp + np.random.normal(0, 5, n_timesteps)
    
    # 传感器3: 压力 (有趋势)
    pressure = 1013 + 0.01 * time + np.random.normal(0, 3, n_timesteps)
    
    # 传感器4: 风速
    wind_speed = 5 + 3 * np.sin(2 * np.pi * time / 24) + np.random.normal(0, 1, n_timesteps)
    
    print(f"原始时间序列长度: {n_timesteps}")
    print(f"传感器数量: 4 (温度, 湿度, 压力, 风速)")
    
    # 转换为监督学习问题
    # 使用过去N个时间步预测未来M个时间步
    lookback = 24  # 使用过去24小时的数据
    forecast = 6   # 预测未来6小时
    
    X_list = []
    y_list = []
    
    for i in range(lookback, n_timesteps - forecast):
        # 输入: 过去24小时的4个传感器数据 (24 * 4 = 96 个特征)
        x_window = np.column_stack([
            temp[i-lookback:i],
            humidity[i-lookback:i], 
            pressure[i-lookback:i],
            wind_speed[i-lookback:i]
        ]).flatten()
        
        # 输出: 未来6小时的温度和湿度 (6 * 2 = 12 个目标)
        y_window = np.column_stack([
            temp[i:i+forecast],
            humidity[i:i+forecast]
        ]).flatten()
        
        X_list.append(x_window)
        y_list.append(y_window)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    print(f"\n转换后数据形状: X={X.shape}, y={y.shape}")
    print(f"输入特征: 过去{lookback}小时的4个传感器数据 ({lookback * 4}维)")
    print(f"输出目标: 未来{forecast}小时的温度和湿度 ({forecast * 2}维)")
    print(f"样本数量: {len(X)}")
    
    # 使用数据训练模型
    config = load_config()
    config['training']['epochs'] = 20
    config['model']['hidden_layers'] = [256, 128, 64]  # 更大的网络处理高维数据
    
    success = train_with_custom_data(X, y, config, "传感器数据预测模型")
    
    return X, y, success


def example_5_image_features():
    """
    示例5: 图像特征数据
    适用场景: 从图像提取的特征进行回归预测
    """
    print("\n" + "="*60)
    print("示例5: 图像特征数据")
    print("="*60)
    
    # 模拟从图像提取的特征 (如CNN特征)
    np.random.seed(42)
    n_samples = 500
    
    # 假设从预训练CNN模型提取的特征向量
    feature_dim = 512  # 常见的特征维度
    
    # 生成图像特征 (模拟ResNet等提取的特征)
    X = np.random.normal(0, 1, (n_samples, feature_dim)).astype(np.float32)
    
    # 添加一些结构化信息
    # 假设前100维是颜色特征，中间200维是纹理特征，后面是形状特征
    X[:, :100] = np.abs(X[:, :100])  # 颜色特征通常为正
    X[:, 100:300] = X[:, 100:300] * 2  # 纹理特征方差更大
    
    # 生成目标变量 (图像质量评分)
    # 目标1: 美学评分 (1-10), 目标2: 技术质量评分 (1-10), 目标3: 情感评分 (-5到5)
    
    # 美学评分主要依赖颜色和构图特征
    aesthetic_score = (
        np.mean(X[:, :50], axis=1) * 2 +  # 颜色特征
        np.mean(X[:, 400:450], axis=1) * 1.5 +  # 构图特征
        np.random.normal(0, 0.5, n_samples)
    )
    aesthetic_score = np.clip(aesthetic_score + 5, 1, 10)  # 缩放到1-10
    
    # 技术质量主要依赖纹理和清晰度特征
    technical_score = (
        np.mean(X[:, 100:200], axis=1) * 1.5 +  # 纹理特征
        np.mean(X[:, 300:350], axis=1) * 2 +    # 清晰度特征
        np.random.normal(0, 0.3, n_samples)
    )
    technical_score = np.clip(technical_score + 5, 1, 10)
    
    # 情感评分依赖多种特征的复杂组合
    emotion_score = (
        np.mean(X[:, 50:100], axis=1) * 1.2 +   # 色彩情感
        np.mean(X[:, 450:500], axis=1) * 0.8 +  # 内容情感
        np.random.normal(0, 0.4, n_samples)
    )
    emotion_score = np.clip(emotion_score, -5, 5)
    
    y = np.column_stack([aesthetic_score, technical_score, emotion_score]).astype(np.float32)
    
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"输入特征: {feature_dim}维图像特征向量")
    print(f"输出目标: 美学评分(1-10), 技术质量(1-10), 情感评分(-5到5)")
    print(f"特征统计:")
    print(f"  特征均值: {X.mean():.3f}, 标准差: {X.std():.3f}")
    print(f"  目标均值: {y.mean(axis=0)}")
    print(f"  目标标准差: {y.std(axis=0)}")
    
    # 使用数据训练模型
    config = load_config()
    config['training']['epochs'] = 20
    config['model']['hidden_layers'] = [256, 128, 64, 32]  # 深层网络处理高维特征
    config['model']['dropout_rate'] = 0.3  # 高维数据需要更多正则化
    
    success = train_with_custom_data(X, y, config, "图像质量评估模型")
    
    return X, y, success


def load_config():
    """加载配置文件"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train_with_custom_data(X: np.ndarray, y: np.ndarray, config: dict, model_name: str) -> bool:
    """使用自定义数据训练模型"""
    try:
        print(f"\n开始训练 {model_name}...")
        
        # 创建数据处理器
        processor = DataProcessor(config)
        processor.load_data_from_arrays(X, y)
        processor.normalize_data()
        
        # 分割数据
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data()
        train_loader, val_loader, test_loader = processor.create_data_loaders(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # 创建模型
        model = create_model_from_config(config, processor.input_dim, processor.output_dim)
        print(f"模型信息: {model.get_model_info()}")
        
        # 训练模型
        trainer = MLPTrainer(model, config)
        history = trainer.train(train_loader, val_loader)
        
        # 评估模型
        evaluator = ModelEvaluator(save_plots=False)  # 不保存图片，避免文件过多
        
        test_pred, test_true = trainer.predict(test_loader)
        test_pred_orig = processor.inverse_transform_predictions(test_pred)
        test_true_orig = processor.inverse_transform_predictions(test_true)
        
        metrics = evaluator.evaluate_model(test_true_orig, test_pred_orig, "test")
        
        print(f"✅ {model_name} 训练成功!")
        print(f"   最终R²分数: {metrics['r2']:.4f}")
        print(f"   RMSE: {metrics['rmse']:.4f}")
        print(f"   MAE: {metrics['mae']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ {model_name} 训练失败: {e}")
        return False


def train_with_processor(processor: DataProcessor, config: dict, model_name: str) -> bool:
    """使用已配置的处理器训练模型"""
    try:
        print(f"\n开始训练 {model_name}...")
        
        processor.normalize_data()
        
        # 分割数据
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data()
        train_loader, val_loader, test_loader = processor.create_data_loaders(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # 创建模型
        model = create_model_from_config(config, processor.input_dim, processor.output_dim)
        
        # 训练模型
        trainer = MLPTrainer(model, config)
        history = trainer.train(train_loader, val_loader)
        
        # 评估模型
        evaluator = ModelEvaluator(save_plots=False)
        
        test_pred, test_true = trainer.predict(test_loader)
        test_pred_orig = processor.inverse_transform_predictions(test_pred)
        test_true_orig = processor.inverse_transform_predictions(test_true)
        
        metrics = evaluator.evaluate_model(test_true_orig, test_pred_orig, "test")
        
        print(f"✅ {model_name} 训练成功!")
        print(f"   最终R²分数: {metrics['r2']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ {model_name} 训练失败: {e}")
        return False


def main():
    """运行所有自定义数据示例"""
    print("🚀 MLP自定义数据集示例演示")
    print("本示例展示如何将不同格式的数据转换为项目可用的格式")
    
    results = []
    
    # 运行所有示例
    examples = [
        ("NumPy数组", example_1_numpy_arrays),
        ("Pandas DataFrame", example_2_pandas_dataframe), 
        ("CSV文件", example_3_csv_file),
        ("时间序列", example_4_time_series_data),
        ("图像特征", example_5_image_features)
    ]
    
    for name, example_func in examples:
        try:
            print(f"\n{'='*20} 运行{name}示例 {'='*20}")
            X, y, success = example_func()
            results.append((name, success, X.shape, y.shape))
        except Exception as e:
            print(f"❌ {name}示例运行失败: {e}")
            results.append((name, False, None, None))
    
    # 总结结果
    print("\n" + "="*60)
    print("📊 示例运行总结")
    print("="*60)
    
    for name, success, x_shape, y_shape in results:
        status = "✅ 成功" if success else "❌ 失败"
        shapes = f"X{x_shape}, y{y_shape}" if x_shape else "N/A"
        print(f"{name:15} | {status:6} | 数据形状: {shapes}")
    
    successful_count = sum(1 for _, success, _, _ in results if success)
    print(f"\n成功运行: {successful_count}/{len(results)} 个示例")
    
    if successful_count > 0:
        print("\n🎉 恭喜！您已经学会了如何在这个项目中使用自定义数据集！")
        print("\n💡 使用建议:")
        print("1. 根据您的数据类型选择合适的示例作为模板")
        print("2. 确保输入数据X为float32类型的2D数组 (n_samples, n_features)")
        print("3. 确保输出数据y为float32类型的2D数组 (n_samples, n_targets)")
        print("4. 根据数据特性调整网络结构和超参数")
        print("5. 使用数据标准化提升训练效果")


if __name__ == "__main__":
    main()