"""
MLP数值预测模型使用示例
展示如何使用各个模块进行数据处理、模型训练和预测
"""

import numpy as np
import yaml
from loguru import logger

from data_processor import DataProcessor, generate_sample_data
from mlp_model import create_model_from_config
from trainer import MLPTrainer
from evaluator import ModelEvaluator


def example_1_basic_usage():
    """示例1: 基本使用流程"""
    print("\n" + "="*60)
    print("示例1: 基本使用流程")
    print("="*60)
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 生成示例数据
    logger.info("生成示例数据...")
    X, y = generate_sample_data(n_samples=1000, n_features=5, n_targets=2, noise=0.1)
    
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
    logger.info(f"模型信息: {model.get_model_info()}")
    
    # 创建训练器
    trainer = MLPTrainer(model, config)
    
    # 训练模型（少量epoch用于演示）
    config['training']['epochs'] = 10
    history = trainer.train(train_loader, val_loader)
    
    # 评估模型
    evaluator = ModelEvaluator(save_plots=False)
    test_pred, test_true = trainer.predict(test_loader)
    
    # 转换回原始尺度
    test_pred_orig = processor.inverse_transform_predictions(test_pred)
    test_true_orig = processor.inverse_transform_predictions(test_true)
    
    # 计算指标
    metrics = evaluator.evaluate_model(test_true_orig, test_pred_orig, "test")
    
    print(f"测试集R²分数: {metrics['r2']:.4f}")
    print("示例1完成!")


def example_2_custom_data():
    """示例2: 使用自定义数据"""
    print("\n" + "="*60)
    print("示例2: 使用自定义数据")
    print("="*60)
    
    # 创建自定义数据（模拟一个简单的数学函数）
    np.random.seed(42)
    n_samples = 500
    
    # 输入特征: x1, x2, x3
    X = np.random.uniform(-2, 2, (n_samples, 3))
    
    # 目标函数: y = x1^2 + 2*x2 + sin(x3) + noise
    y = (X[:, 0]**2 + 2*X[:, 1] + np.sin(X[:, 2]) + 
         0.1 * np.random.randn(n_samples)).reshape(-1, 1)
    
    logger.info(f"自定义数据: X形状={X.shape}, y形状={y.shape}")
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 修改配置以适应数据
    config['training']['epochs'] = 20
    config['model']['hidden_layers'] = [64, 32]
    
    # 数据处理
    processor = DataProcessor(config)
    processor.load_data_from_arrays(X, y)
    processor.normalize_data()
    
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data()
    train_loader, val_loader, test_loader = processor.create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # 创建和训练模型
    model = create_model_from_config(config, processor.input_dim, processor.output_dim)
    trainer = MLPTrainer(model, config)
    
    history = trainer.train(train_loader, val_loader)
    
    # 评估
    evaluator = ModelEvaluator(save_plots=False)
    test_pred, test_true = trainer.predict(test_loader)
    
    test_pred_orig = processor.inverse_transform_predictions(test_pred)
    test_true_orig = processor.inverse_transform_predictions(test_true)
    
    metrics = evaluator.evaluate_model(test_true_orig, test_pred_orig, "test")
    
    print(f"自定义数据测试集R²分数: {metrics['r2']:.4f}")
    print("示例2完成!")


def example_3_model_comparison():
    """示例3: 不同模型配置的比较"""
    print("\n" + "="*60)
    print("示例3: 不同模型配置的比较")
    print("="*60)
    
    # 生成数据
    X, y = generate_sample_data(n_samples=800, n_features=8, n_targets=1, noise=0.15)
    
    # 基础配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    base_config['training']['epochs'] = 15
    
    # 不同的模型配置
    model_configs = [
        {
            'name': '小模型',
            'hidden_layers': [32, 16],
            'dropout_rate': 0.1
        },
        {
            'name': '中等模型',
            'hidden_layers': [64, 32, 16],
            'dropout_rate': 0.2
        },
        {
            'name': '大模型',
            'hidden_layers': [128, 64, 32, 16],
            'dropout_rate': 0.3
        }
    ]
    
    results = []
    
    for model_config in model_configs:
        logger.info(f"训练 {model_config['name']}...")
        
        # 修改配置
        import copy
        config = copy.deepcopy(base_config)
        config['model']['hidden_layers'] = model_config['hidden_layers']
        config['model']['dropout_rate'] = model_config['dropout_rate']
        
        # 数据处理
        processor = DataProcessor(config)
        processor.load_data_from_arrays(X, y)
        processor.normalize_data()
        
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data()
        train_loader, val_loader, test_loader = processor.create_data_loaders(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # 训练模型
        model = create_model_from_config(config, processor.input_dim, processor.output_dim)
        trainer = MLPTrainer(model, config)
        
        history = trainer.train(train_loader, val_loader)
        
        # 评估
        test_pred, test_true = trainer.predict(test_loader)
        test_pred_orig = processor.inverse_transform_predictions(test_pred)
        test_true_orig = processor.inverse_transform_predictions(test_true)
        
        evaluator = ModelEvaluator(save_plots=False)
        metrics = evaluator.evaluate_model(test_true_orig, test_pred_orig, "test")
        
        results.append({
            'name': model_config['name'],
            'r2': metrics['r2'],
            'rmse': metrics['rmse'],
            'parameters': model.get_model_info()['total_parameters']
        })
    
    # 打印比较结果
    print("\n模型比较结果:")
    print("-" * 60)
    print(f"{'模型名称':<10} {'参数数量':<10} {'R²分数':<10} {'RMSE':<10}")
    print("-" * 60)
    for result in results:
        print(f"{result['name']:<10} {result['parameters']:<10} {result['r2']:<10.4f} {result['rmse']:<10.4f}")
    
    print("示例3完成!")


def example_4_prediction_demo():
    """示例4: 预测演示"""
    print("\n" + "="*60)
    print("示例4: 预测演示")
    print("="*60)
    
    # 生成训练数据
    X_train_full, y_train_full = generate_sample_data(n_samples=600, n_features=4, n_targets=2)
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    config['training']['epochs'] = 15
    
    # 训练模型
    processor = DataProcessor(config)
    processor.load_data_from_arrays(X_train_full, y_train_full)
    processor.normalize_data()
    
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data()
    train_loader, val_loader, test_loader = processor.create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    model = create_model_from_config(config, processor.input_dim, processor.output_dim)
    trainer = MLPTrainer(model, config)
    
    logger.info("训练模型...")
    history = trainer.train(train_loader, val_loader)
    
    # 生成新的预测数据
    logger.info("生成新数据进行预测...")
    X_new = np.random.randn(5, 4).astype(np.float32)
    
    # 使用训练好的模型进行预测
    predictions_scaled = trainer.predict_single(X_new)
    predictions = processor.inverse_transform_predictions(predictions_scaled)
    
    print("\n预测结果:")
    print("-" * 40)
    for i, (input_data, pred) in enumerate(zip(X_new, predictions)):
        print(f"输入 {i+1}: {input_data}")
        print(f"预测 {i+1}: {pred}")
        print("-" * 40)
    
    print("示例4完成!")


def main():
    """运行所有示例"""
    logger.info("开始运行MLP数值预测模型示例...")
    
    try:
        example_1_basic_usage()
        example_2_custom_data()
        example_3_model_comparison()
        example_4_prediction_demo()
        
        print("\n" + "="*60)
        print("所有示例运行完成!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"运行示例时出错: {e}")
        raise


if __name__ == "__main__":
    main()