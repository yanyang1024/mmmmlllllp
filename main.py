"""
MLP数值预测主程序
整合所有模块，提供完整的训练和预测流程
"""

import os
import yaml
import numpy as np
import torch
from loguru import logger
import argparse
from typing import Optional, Tuple

from data_processor import DataProcessor, generate_sample_data
from mlp_model import create_model_from_config
from trainer import MLPTrainer
from evaluator import ModelEvaluator


def setup_logging(config: dict) -> None:
    """设置日志系统"""
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')
    
    # 移除默认的logger
    logger.remove()
    
    # 添加控制台输出
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # 如果配置了保存日志
    if log_config.get('save_logs', False):
        log_dir = log_config.get('log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        logger.add(
            os.path.join(log_dir, "mlp_training.log"),
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB"
        )


def load_config(config_path: str = "config.yaml") -> dict:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"配置文件加载成功: {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"配置文件未找到: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"配置文件格式错误: {e}")
        raise


def prepare_data(config: dict, 
                data_source: str = "sample",
                csv_path: Optional[str] = None,
                target_columns: Optional[list] = None,
                feature_columns: Optional[list] = None,
                X: Optional[np.ndarray] = None,
                y: Optional[np.ndarray] = None) -> Tuple[DataProcessor, tuple]:
    """
    准备数据
    
    Args:
        config: 配置字典
        data_source: 数据源类型 ("sample", "csv", "arrays")
        csv_path: CSV文件路径
        target_columns: 目标列名
        feature_columns: 特征列名
        X: 输入特征数组
        y: 目标值数组
        
    Returns:
        数据处理器和数据加载器元组
    """
    logger.info("开始准备数据...")
    
    # 创建数据处理器
    processor = DataProcessor(config)
    
    # 根据数据源加载数据
    if data_source == "sample":
        logger.info("生成示例数据...")
        X_sample, y_sample = generate_sample_data(
            n_samples=1000,
            n_features=10,
            n_targets=3,
            noise=0.1,
            random_seed=config['data']['random_seed']
        )
        processor.load_data_from_arrays(X_sample, y_sample)
        
    elif data_source == "csv":
        if csv_path is None or target_columns is None:
            raise ValueError("使用CSV数据源时必须提供csv_path和target_columns")
        processor.load_data_from_csv(csv_path, target_columns, feature_columns)
        
    elif data_source == "arrays":
        if X is None or y is None:
            raise ValueError("使用数组数据源时必须提供X和y")
        processor.load_data_from_arrays(X, y)
        
    else:
        raise ValueError(f"不支持的数据源类型: {data_source}")
    
    # 数据预处理
    processor.normalize_data()
    
    # 分割数据
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data()
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = processor.create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    logger.info("数据准备完成")
    
    return processor, (train_loader, val_loader, test_loader, X_test, y_test)


def train_model(config: dict, 
               processor: DataProcessor,
               data_loaders: tuple,
               save_dir: str = "models") -> MLPTrainer:
    """
    训练模型
    
    Args:
        config: 配置字典
        processor: 数据处理器
        data_loaders: 数据加载器元组
        save_dir: 模型保存目录
        
    Returns:
        训练好的训练器
    """
    train_loader, val_loader, test_loader, X_test, y_test = data_loaders
    
    logger.info("开始训练模型...")
    
    # 创建模型
    model = create_model_from_config(config, processor.input_dim, processor.output_dim)
    
    # 打印模型信息
    model_info = model.get_model_info()
    logger.info(f"模型信息: {model_info}")
    
    # 创建训练器
    trainer = MLPTrainer(model, config)
    
    # 开始训练
    history = trainer.train(train_loader, val_loader, save_dir=save_dir)
    
    # 保存完整模型
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(os.path.join(save_dir, "final_model.pth"))
    
    # 保存数据处理器
    processor.save_scalers(save_dir)
    
    logger.info("模型训练完成")
    
    return trainer


def evaluate_model(trainer: MLPTrainer,
                  processor: DataProcessor,
                  data_loaders: tuple,
                  save_plots: bool = True) -> dict:
    """
    评估模型
    
    Args:
        trainer: 训练器
        processor: 数据处理器
        data_loaders: 数据加载器元组
        save_plots: 是否保存图表
        
    Returns:
        评估结果字典
    """
    train_loader, val_loader, test_loader, X_test, y_test = data_loaders
    
    logger.info("开始评估模型...")
    
    # 创建评估器
    evaluator = ModelEvaluator(save_plots=save_plots)
    
    # 在各个数据集上进行预测
    train_pred, train_true = trainer.predict(train_loader)
    val_pred, val_true = trainer.predict(val_loader)
    test_pred, test_true = trainer.predict(test_loader)
    
    # 转换回原始尺度
    train_pred_orig = processor.inverse_transform_predictions(train_pred)
    train_true_orig = processor.inverse_transform_predictions(train_true)
    val_pred_orig = processor.inverse_transform_predictions(val_pred)
    val_true_orig = processor.inverse_transform_predictions(val_true)
    test_pred_orig = processor.inverse_transform_predictions(test_pred)
    test_true_orig = processor.inverse_transform_predictions(test_true)
    
    # 计算评估指标
    train_metrics = evaluator.evaluate_model(train_true_orig, train_pred_orig, "train")
    val_metrics = evaluator.evaluate_model(val_true_orig, val_pred_orig, "validation")
    test_metrics = evaluator.evaluate_model(test_true_orig, test_pred_orig, "test")
    
    # 绘制图表
    if save_plots:
        evaluator.plot_predictions_vs_actual(test_true_orig, test_pred_orig, "test")
        evaluator.plot_residuals(test_true_orig, test_pred_orig, "test")
        evaluator.plot_training_history(trainer.history)
    
    # 创建评估报告
    model_info = trainer.model.get_model_info()
    evaluator.create_evaluation_report(
        train_metrics, val_metrics, test_metrics, model_info
    )
    
    logger.info("模型评估完成")
    
    return {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'model_info': model_info
    }


def predict_new_data(model_path: str,
                    scaler_dir: str,
                    X_new: np.ndarray,
                    config_path: str = "config.yaml") -> np.ndarray:
    """
    使用训练好的模型对新数据进行预测
    
    Args:
        model_path: 模型文件路径
        scaler_dir: 标准化器目录
        X_new: 新的输入数据
        config_path: 配置文件路径
        
    Returns:
        预测结果
    """
    logger.info("开始预测新数据...")
    
    # 加载配置
    config = load_config(config_path)
    
    # 创建数据处理器并加载标准化器
    processor = DataProcessor(config)
    processor.load_scalers(scaler_dir)
    
    # 标准化输入数据
    if processor.scaler_X is not None:
        X_new_scaled = processor.scaler_X.transform(X_new)
    else:
        X_new_scaled = X_new
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = checkpoint['model_config']
    
    # 重建模型
    from mlp_model import MLPModel
    model = MLPModel(
        input_dim=model_config['input_dim'],
        output_dim=model_config['output_dim'],
        hidden_layers=model_config['hidden_layers'],
        activation=model_config['activation'],
        dropout_rate=model_config['dropout_rate']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建训练器进行预测
    trainer = MLPTrainer(model, config)
    predictions_scaled = trainer.predict_single(X_new_scaled)
    
    # 转换回原始尺度
    predictions = processor.inverse_transform_predictions(predictions_scaled)
    
    logger.info(f"预测完成，预测了 {len(X_new)} 个样本")
    
    return predictions


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MLP数值预测模型')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train', help='运行模式')
    parser.add_argument('--model_path', type=str, help='模型文件路径（预测模式）')
    parser.add_argument('--data_path', type=str, help='数据文件路径')
    parser.add_argument('--save_dir', type=str, default='models', help='模型保存目录')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    setup_logging(config)
    
    if args.mode == 'train':
        logger.info("=== 开始训练模式 ===")
        
        # 准备数据
        processor, data_loaders = prepare_data(config, data_source="sample")
        
        # 训练模型
        trainer = train_model(config, processor, data_loaders, args.save_dir)
        
        # 评估模型
        evaluation_results = evaluate_model(trainer, processor, data_loaders)
        
        # 打印训练摘要
        summary = trainer.get_training_summary()
        logger.info("=== 训练摘要 ===")
        for key, value in summary.items():
            logger.info(f"{key}: {value}")
            
        logger.info("=== 训练完成 ===")
        
    elif args.mode == 'predict':
        if args.model_path is None:
            logger.error("预测模式需要指定模型路径")
            return
            
        logger.info("=== 开始预测模式 ===")
        
        # 生成一些示例数据进行预测
        X_new = np.random.randn(5, 10).astype(np.float32)
        
        # 进行预测
        predictions = predict_new_data(
            args.model_path,
            args.save_dir,
            X_new,
            args.config
        )
        
        logger.info("预测结果:")
        for i, pred in enumerate(predictions):
            logger.info(f"样本 {i+1}: {pred}")
            
        logger.info("=== 预测完成 ===")


if __name__ == "__main__":
    main()