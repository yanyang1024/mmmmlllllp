"""
NaN值处理指南和工具
提供多种处理缺失值的策略和实现方法
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, Optional, List
from loguru import logger


class NaNHandler:
    """NaN值处理器，提供多种缺失值处理策略"""
    
    def __init__(self):
        self.imputers = {}
        self.nan_info = {}
        
    def analyze_nan_pattern(self, X: np.ndarray, y: np.ndarray, 
                           feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        分析NaN值的分布模式
        
        Args:
            X: 输入特征数组
            y: 目标值数组
            feature_names: 特征名称列表
            
        Returns:
            NaN分析结果字典
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
        # 统计每个特征的NaN数量
        nan_counts_X = np.isnan(X).sum(axis=0)
        nan_ratios_X = nan_counts_X / X.shape[0]
        
        # 统计目标值的NaN数量
        nan_counts_y = np.isnan(y).sum(axis=0) if y.ndim > 1 else np.isnan(y).sum()
        nan_ratios_y = nan_counts_y / y.shape[0]
        
        # 统计每个样本的NaN数量
        nan_per_sample = np.isnan(X).sum(axis=1)
        
        # 完全缺失的样本数量
        complete_missing_samples = np.sum(nan_per_sample == X.shape[1])
        
        # 完全完整的样本数量
        complete_samples = np.sum(nan_per_sample == 0)
        
        analysis = {
            'total_samples': X.shape[0],
            'total_features': X.shape[1],
            'feature_names': feature_names,
            'nan_counts_per_feature': dict(zip(feature_names, nan_counts_X)),
            'nan_ratios_per_feature': dict(zip(feature_names, nan_ratios_X)),
            'nan_counts_targets': nan_counts_y,
            'nan_ratios_targets': nan_ratios_y,
            'samples_with_nan': np.sum(nan_per_sample > 0),
            'complete_samples': complete_samples,
            'complete_missing_samples': complete_missing_samples,
            'max_nan_per_sample': np.max(nan_per_sample),
            'avg_nan_per_sample': np.mean(nan_per_sample),
            'features_with_high_nan': [name for name, ratio in zip(feature_names, nan_ratios_X) if ratio > 0.5]
        }
        
        self.nan_info = analysis
        return analysis
    
    def print_nan_summary(self, analysis: Dict[str, Any]):
        """打印NaN分析摘要"""
        print("=" * 60)
        print("🔍 NaN值分析报告")
        print("=" * 60)
        
        print(f"📊 数据概况:")
        print(f"  总样本数: {analysis['total_samples']}")
        print(f"  总特征数: {analysis['total_features']}")
        print(f"  完整样本数: {analysis['complete_samples']} ({analysis['complete_samples']/analysis['total_samples']*100:.1f}%)")
        print(f"  含缺失值样本数: {analysis['samples_with_nan']} ({analysis['samples_with_nan']/analysis['total_samples']*100:.1f}%)")
        
        print(f"\n🎯 目标值缺失情况:")
        if isinstance(analysis['nan_counts_targets'], np.ndarray):
            for i, (count, ratio) in enumerate(zip(analysis['nan_counts_targets'], analysis['nan_ratios_targets'])):
                print(f"  目标{i+1}: {count}个缺失 ({ratio*100:.1f}%)")
        else:
            print(f"  缺失数量: {analysis['nan_counts_targets']} ({analysis['nan_ratios_targets']*100:.1f}%)")
        
        print(f"\n📈 特征缺失情况 (前10个最严重的):")
        sorted_features = sorted(analysis['nan_ratios_per_feature'].items(), 
                               key=lambda x: x[1], reverse=True)[:10]
        for name, ratio in sorted_features:
            if ratio > 0:
                count = analysis['nan_counts_per_feature'][name]
                print(f"  {name}: {count}个缺失 ({ratio*100:.1f}%)")
        
        if analysis['features_with_high_nan']:
            print(f"\n⚠️  高缺失率特征 (>50%): {len(analysis['features_with_high_nan'])}个")
            for name in analysis['features_with_high_nan'][:5]:
                ratio = analysis['nan_ratios_per_feature'][name]
                print(f"    {name}: {ratio*100:.1f}%")
    
    def visualize_nan_pattern(self, X: np.ndarray, feature_names: Optional[List[str]] = None,
                             save_path: Optional[str] = None):
        """可视化NaN值分布模式"""
        if feature_names is None:
            feature_names = [f'F{i}' for i in range(X.shape[1])]
            
        # 创建NaN掩码
        nan_mask = np.isnan(X)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 每个特征的缺失率
        nan_ratios = nan_mask.mean(axis=0)
        axes[0, 0].bar(range(len(feature_names)), nan_ratios)
        axes[0, 0].set_title('每个特征的缺失率')
        axes[0, 0].set_xlabel('特征索引')
        axes[0, 0].set_ylabel('缺失率')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 每个样本的缺失数量分布
        nan_per_sample = nan_mask.sum(axis=1)
        axes[0, 1].hist(nan_per_sample, bins=min(50, X.shape[1]), alpha=0.7)
        axes[0, 1].set_title('每个样本的缺失特征数量分布')
        axes[0, 1].set_xlabel('缺失特征数量')
        axes[0, 1].set_ylabel('样本数量')
        
        # 3. NaN模式热力图 (取前100个样本和前20个特征)
        sample_size = min(100, X.shape[0])
        feature_size = min(20, X.shape[1])
        
        axes[1, 0].imshow(nan_mask[:sample_size, :feature_size], 
                         cmap='RdYlBu_r', aspect='auto')
        axes[1, 0].set_title(f'NaN模式热力图 (前{sample_size}样本, 前{feature_size}特征)')
        axes[1, 0].set_xlabel('特征索引')
        axes[1, 0].set_ylabel('样本索引')
        
        # 4. 缺失率分布
        axes[1, 1].hist(nan_ratios, bins=20, alpha=0.7)
        axes[1, 1].set_title('特征缺失率分布')
        axes[1, 1].set_xlabel('缺失率')
        axes[1, 1].set_ylabel('特征数量')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"NaN分析图表已保存到: {save_path}")
        
        plt.show()
    
    def strategy_1_remove_samples(self, X: np.ndarray, y: np.ndarray, 
                                 threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        策略1: 删除含有过多NaN的样本
        
        Args:
            X: 输入特征
            y: 目标值
            threshold: 缺失率阈值，超过此比例的样本将被删除
            
        Returns:
            清理后的X, y
        """
        nan_ratio_per_sample = np.isnan(X).sum(axis=1) / X.shape[1]
        mask = nan_ratio_per_sample <= threshold
        
        X_clean = X[mask]
        y_clean = y[mask]
        
        removed_count = np.sum(~mask)
        logger.info(f"策略1: 删除了{removed_count}个样本 (缺失率>{threshold*100:.1f}%)")
        logger.info(f"剩余样本数: {X_clean.shape[0]} / {X.shape[0]}")
        
        return X_clean, y_clean
    
    def strategy_2_remove_features(self, X: np.ndarray, y: np.ndarray,
                                  threshold: float = 0.7,
                                  feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        策略2: 删除含有过多NaN的特征
        
        Args:
            X: 输入特征
            y: 目标值
            threshold: 缺失率阈值，超过此比例的特征将被删除
            feature_names: 特征名称列表
            
        Returns:
            清理后的X, y, 保留的特征名称
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
        nan_ratio_per_feature = np.isnan(X).mean(axis=0)
        mask = nan_ratio_per_feature <= threshold
        
        X_clean = X[:, mask]
        remaining_features = [name for name, keep in zip(feature_names, mask) if keep]
        
        removed_count = np.sum(~mask)
        logger.info(f"策略2: 删除了{removed_count}个特征 (缺失率>{threshold*100:.1f}%)")
        logger.info(f"剩余特征数: {X_clean.shape[1]} / {X.shape[1]}")
        
        return X_clean, y, remaining_features
    
    def strategy_3_simple_imputation(self, X: np.ndarray, y: np.ndarray,
                                   strategy: str = 'mean') -> Tuple[np.ndarray, np.ndarray]:
        """
        策略3: 简单插值填充
        
        Args:
            X: 输入特征
            y: 目标值
            strategy: 插值策略 ('mean', 'median', 'most_frequent', 'constant')
            
        Returns:
            填充后的X, y
        """
        # 处理输入特征
        imputer_X = SimpleImputer(strategy=strategy)
        X_imputed = imputer_X.fit_transform(X)
        self.imputers['X_simple'] = imputer_X
        
        # 处理目标值
        if np.isnan(y).any():
            imputer_y = SimpleImputer(strategy=strategy if strategy != 'most_frequent' else 'mean')
            y_imputed = imputer_y.fit_transform(y.reshape(-1, 1) if y.ndim == 1 else y)
            if y.ndim == 1:
                y_imputed = y_imputed.ravel()
            self.imputers['y_simple'] = imputer_y
        else:
            y_imputed = y
        
        logger.info(f"策略3: 使用{strategy}策略进行简单插值填充")
        
        return X_imputed, y_imputed
    
    def strategy_4_knn_imputation(self, X: np.ndarray, y: np.ndarray,
                                 n_neighbors: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        策略4: KNN插值填充
        
        Args:
            X: 输入特征
            y: 目标值
            n_neighbors: KNN的邻居数量
            
        Returns:
            填充后的X, y
        """
        # 处理输入特征
        imputer_X = KNNImputer(n_neighbors=n_neighbors)
        X_imputed = imputer_X.fit_transform(X)
        self.imputers['X_knn'] = imputer_X
        
        # 处理目标值
        if np.isnan(y).any():
            imputer_y = KNNImputer(n_neighbors=n_neighbors)
            y_imputed = imputer_y.fit_transform(y.reshape(-1, 1) if y.ndim == 1 else y)
            if y.ndim == 1:
                y_imputed = y_imputed.ravel()
            self.imputers['y_knn'] = imputer_y
        else:
            y_imputed = y
        
        logger.info(f"策略4: 使用KNN插值填充 (k={n_neighbors})")
        
        return X_imputed, y_imputed
    
    def strategy_5_iterative_imputation(self, X: np.ndarray, y: np.ndarray,
                                       max_iter: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        策略5: 迭代插值填充 (MICE算法)
        
        Args:
            X: 输入特征
            y: 目标值
            max_iter: 最大迭代次数
            
        Returns:
            填充后的X, y
        """
        # 处理输入特征
        imputer_X = IterativeImputer(max_iter=max_iter, random_state=42)
        X_imputed = imputer_X.fit_transform(X)
        self.imputers['X_iterative'] = imputer_X
        
        # 处理目标值
        if np.isnan(y).any():
            imputer_y = IterativeImputer(max_iter=max_iter, random_state=42)
            y_imputed = imputer_y.fit_transform(y.reshape(-1, 1) if y.ndim == 1 else y)
            if y.ndim == 1:
                y_imputed = y_imputed.ravel()
            self.imputers['y_iterative'] = imputer_y
        else:
            y_imputed = y
        
        logger.info(f"策略5: 使用迭代插值填充 (最大迭代{max_iter}次)")
        
        return X_imputed, y_imputed
    
    def strategy_6_hybrid_approach(self, X: np.ndarray, y: np.ndarray,
                                  high_missing_threshold: float = 0.7,
                                  sample_missing_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        策略6: 混合方法
        1. 删除高缺失率特征
        2. 删除高缺失率样本
        3. 对剩余数据进行KNN插值
        
        Args:
            X: 输入特征
            y: 目标值
            high_missing_threshold: 高缺失率特征阈值
            sample_missing_threshold: 高缺失率样本阈值
            
        Returns:
            处理后的X, y
        """
        logger.info("策略6: 混合方法处理")
        
        original_shape = X.shape
        
        # 步骤1: 删除高缺失率特征
        feature_nan_ratio = np.isnan(X).mean(axis=0)
        feature_mask = feature_nan_ratio <= high_missing_threshold
        X_step1 = X[:, feature_mask]
        
        removed_features = np.sum(~feature_mask)
        logger.info(f"  步骤1: 删除{removed_features}个高缺失率特征")
        
        # 步骤2: 删除高缺失率样本
        sample_nan_ratio = np.isnan(X_step1).sum(axis=1) / X_step1.shape[1]
        sample_mask = sample_nan_ratio <= sample_missing_threshold
        X_step2 = X_step1[sample_mask]
        y_step2 = y[sample_mask]
        
        removed_samples = np.sum(~sample_mask)
        logger.info(f"  步骤2: 删除{removed_samples}个高缺失率样本")
        
        # 步骤3: KNN插值填充剩余缺失值
        if np.isnan(X_step2).any():
            X_final, y_final = self.strategy_4_knn_imputation(X_step2, y_step2)
            logger.info(f"  步骤3: 对剩余缺失值进行KNN插值")
        else:
            X_final, y_final = X_step2, y_step2
            logger.info(f"  步骤3: 无需插值，数据已完整")
        
        logger.info(f"混合方法完成: {original_shape} -> {X_final.shape}")
        
        return X_final, y_final
    
    def compare_strategies(self, X: np.ndarray, y: np.ndarray,
                          strategies: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        比较不同NaN处理策略的效果
        
        Args:
            X: 输入特征
            y: 目标值
            strategies: 要比较的策略列表
            
        Returns:
            各策略的比较结果
        """
        if strategies is None:
            strategies = ['remove_samples', 'simple_mean', 'knn', 'iterative', 'hybrid']
        
        results = {}
        
        for strategy in strategies:
            try:
                if strategy == 'remove_samples':
                    X_proc, y_proc = self.strategy_1_remove_samples(X.copy(), y.copy())
                elif strategy == 'simple_mean':
                    X_proc, y_proc = self.strategy_3_simple_imputation(X.copy(), y.copy(), 'mean')
                elif strategy == 'simple_median':
                    X_proc, y_proc = self.strategy_3_simple_imputation(X.copy(), y.copy(), 'median')
                elif strategy == 'knn':
                    X_proc, y_proc = self.strategy_4_knn_imputation(X.copy(), y.copy())
                elif strategy == 'iterative':
                    X_proc, y_proc = self.strategy_5_iterative_imputation(X.copy(), y.copy())
                elif strategy == 'hybrid':
                    X_proc, y_proc = self.strategy_6_hybrid_approach(X.copy(), y.copy())
                else:
                    continue
                
                # 计算处理后的统计信息
                results[strategy] = {
                    'final_shape': X_proc.shape,
                    'samples_retained': X_proc.shape[0] / X.shape[0],
                    'features_retained': X_proc.shape[1] / X.shape[1],
                    'has_nan': np.isnan(X_proc).any() or np.isnan(y_proc).any(),
                    'data_mean': np.nanmean(X_proc),
                    'data_std': np.nanstd(X_proc)
                }
                
            except Exception as e:
                results[strategy] = {'error': str(e)}
        
        return results
    
    def recommend_strategy(self, analysis: Dict[str, Any]) -> str:
        """
        根据数据分析结果推荐最佳策略
        
        Args:
            analysis: NaN分析结果
            
        Returns:
            推荐的策略名称
        """
        total_samples = analysis['total_samples']
        complete_samples = analysis['complete_samples']
        complete_ratio = complete_samples / total_samples
        
        high_missing_features = len(analysis['features_with_high_nan'])
        total_features = analysis['total_features']
        
        avg_nan_per_sample = analysis['avg_nan_per_sample']
        
        print("\n🤖 策略推荐分析:")
        print(f"  完整样本比例: {complete_ratio*100:.1f}%")
        print(f"  高缺失特征数: {high_missing_features}/{total_features}")
        print(f"  平均每样本缺失特征数: {avg_nan_per_sample:.1f}")
        
        if complete_ratio >= 0.7:
            recommendation = "remove_samples"
            reason = "完整样本比例较高，建议删除含缺失值的样本"
        elif high_missing_features > total_features * 0.3:
            recommendation = "hybrid"
            reason = "高缺失特征较多，建议使用混合方法"
        elif avg_nan_per_sample <= 2:
            recommendation = "knn"
            reason = "缺失值较少且分散，KNN插值效果较好"
        elif total_samples >= 1000:
            recommendation = "iterative"
            reason = "样本量充足，可使用迭代插值获得更好效果"
        else:
            recommendation = "simple_mean"
            reason = "样本量较少，使用简单均值插值较为稳妥"
        
        print(f"\n💡 推荐策略: {recommendation}")
        print(f"   推荐理由: {reason}")
        
        return recommendation


def demonstrate_nan_handling():
    """演示NaN处理的完整流程"""
    print("🚀 NaN值处理演示")
    print("=" * 60)
    
    # 创建含有NaN的示例数据
    np.random.seed(42)
    n_samples, n_features = 1000, 15
    
    # 生成基础数据
    X_clean = np.random.randn(n_samples, n_features).astype(np.float32)
    y_clean = (X_clean[:, :3].sum(axis=1) + np.random.randn(n_samples) * 0.1).astype(np.float32)
    
    # 人为引入NaN值
    X = X_clean.copy()
    y = y_clean.copy()
    
    # 在X中随机引入NaN (不同特征有不同的缺失率)
    for i in range(n_features):
        missing_rate = np.random.uniform(0.05, 0.4)  # 5%-40%的缺失率
        n_missing = int(n_samples * missing_rate)
        missing_indices = np.random.choice(n_samples, n_missing, replace=False)
        X[missing_indices, i] = np.nan
    
    # 在y中引入少量NaN
    y_missing_indices = np.random.choice(n_samples, int(n_samples * 0.02), replace=False)
    y[y_missing_indices] = np.nan
    
    print(f"创建了含NaN的示例数据: X{X.shape}, y{y.shape}")
    
    # 创建NaN处理器
    handler = NaNHandler()
    
    # 分析NaN模式
    analysis = handler.analyze_nan_pattern(X, y)
    handler.print_nan_summary(analysis)
    
    # 可视化NaN模式
    handler.visualize_nan_pattern(X, save_path="nan_pattern_analysis.png")
    
    # 获取推荐策略
    recommended = handler.recommend_strategy(analysis)
    
    # 比较不同策略
    print("\n📊 策略比较结果:")
    comparison = handler.compare_strategies(X, y)
    
    for strategy, result in comparison.items():
        if 'error' in result:
            print(f"  {strategy}: ❌ 错误 - {result['error']}")
        else:
            print(f"  {strategy}:")
            print(f"    最终形状: {result['final_shape']}")
            print(f"    样本保留率: {result['samples_retained']*100:.1f}%")
            print(f"    特征保留率: {result['features_retained']*100:.1f}%")
            print(f"    仍有NaN: {'是' if result['has_nan'] else '否'}")
    
    # 应用推荐策略
    print(f"\n🎯 应用推荐策略: {recommended}")
    
    if recommended == 'remove_samples':
        X_final, y_final = handler.strategy_1_remove_samples(X, y)
    elif recommended == 'simple_mean':
        X_final, y_final = handler.strategy_3_simple_imputation(X, y, 'mean')
    elif recommended == 'knn':
        X_final, y_final = handler.strategy_4_knn_imputation(X, y)
    elif recommended == 'iterative':
        X_final, y_final = handler.strategy_5_iterative_imputation(X, y)
    elif recommended == 'hybrid':
        X_final, y_final = handler.strategy_6_hybrid_approach(X, y)
    
    print(f"✅ 处理完成!")
    print(f"   原始数据: X{X.shape}, y{y.shape}")
    print(f"   处理后: X{X_final.shape}, y{y_final.shape}")
    print(f"   是否还有NaN: {np.isnan(X_final).any() or np.isnan(y_final).any()}")
    
    return X_final, y_final, handler


def create_nan_data_template():
    """创建处理含NaN数据的模板代码"""
    template = '''
# 处理含NaN值数据的完整模板
import numpy as np
import yaml
from nan_handling_guide import NaNHandler
from data_processor import DataProcessor
from mlp_model import create_model_from_config
from trainer import MLPTrainer

def train_with_nan_data():
    """处理含NaN数据并训练模型的完整流程"""
    
    # 1. 加载您的含NaN数据
    # ================================
    # TODO: 替换为您的实际数据加载代码
    X = your_data_with_nan  # 含NaN的输入特征
    y = your_targets_with_nan  # 可能含NaN的目标值
    # ================================
    
    # 2. 创建NaN处理器并分析数据
    handler = NaNHandler()
    analysis = handler.analyze_nan_pattern(X, y)
    handler.print_nan_summary(analysis)
    
    # 3. 获取推荐策略
    recommended_strategy = handler.recommend_strategy(analysis)
    
    # 4. 应用处理策略
    if recommended_strategy == 'remove_samples':
        X_clean, y_clean = handler.strategy_1_remove_samples(X, y, threshold=0.5)
    elif recommended_strategy == 'simple_mean':
        X_clean, y_clean = handler.strategy_3_simple_imputation(X, y, 'mean')
    elif recommended_strategy == 'knn':
        X_clean, y_clean = handler.strategy_4_knn_imputation(X, y, n_neighbors=5)
    elif recommended_strategy == 'iterative':
        X_clean, y_clean = handler.strategy_5_iterative_imputation(X, y)
    elif recommended_strategy == 'hybrid':
        X_clean, y_clean = handler.strategy_6_hybrid_approach(X, y)
    else:
        # 默认使用混合方法
        X_clean, y_clean = handler.strategy_6_hybrid_approach(X, y)
    
    print(f"NaN处理完成: {X.shape} -> {X_clean.shape}")
    
    # 5. 验证数据清洁度
    assert not np.isnan(X_clean).any(), "输入特征仍有NaN值"
    assert not np.isnan(y_clean).any(), "目标值仍有NaN值"
    print("✅ 数据验证通过，无NaN值")
    
    # 6. 正常的模型训练流程
    config = yaml.safe_load(open('config.yaml', 'r', encoding='utf-8'))
    
    processor = DataProcessor(config)
    processor.load_data_from_arrays(X_clean, y_clean)
    processor.normalize_data()
    
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data()
    train_loader, val_loader, test_loader = processor.create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    model = create_model_from_config(config, processor.input_dim, processor.output_dim)
    trainer = MLPTrainer(model, config)
    
    print("开始训练...")
    history = trainer.train(train_loader, val_loader)
    
    # 7. 评估模型
    from evaluator import ModelEvaluator
    evaluator = ModelEvaluator()
    test_pred, test_true = trainer.predict(test_loader)
    test_pred_orig = processor.inverse_transform_predictions(test_pred)
    test_true_orig = processor.inverse_transform_predictions(test_true)
    
    metrics = evaluator.evaluate_model(test_true_orig, test_pred_orig)
    print(f"训练完成！R²分数: {metrics['r2']:.4f}")
    
    return trainer, processor, metrics, handler

if __name__ == "__main__":
    train_with_nan_data()
'''
    
    with open('nan_data_template.py', 'w', encoding='utf-8') as f:
        f.write(template)
    
    print("📝 已创建NaN数据处理模板: nan_data_template.py")


if __name__ == "__main__":
    # 运行演示
    demonstrate_nan_handling()
    
    # 创建模板
    create_nan_data_template()
    
    print("\n🎉 NaN处理指南演示完成！")
    print("💡 使用建议:")
    print("1. 首先分析您的数据NaN分布模式")
    print("2. 根据推荐选择合适的处理策略")
    print("3. 验证处理后的数据质量")
    print("4. 使用处理后的数据进行正常训练")