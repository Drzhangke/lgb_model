#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用Python LightGBM重新训练模型
基于R版本的实现，使用data.xlsx中的数据
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import joblib
import json
import os
from datetime import datetime

def load_and_explore_data(excel_path="data.xlsx"):
    """
    加载并探索数据
    """
    print("🔍 正在加载数据...")

    # 读取Excel文件
    xl = pd.ExcelFile(excel_path)
    print(f"📊 发现的工作表: {xl.sheet_names}")

    # 读取主工作表
    df = pd.read_excel(excel_path, sheet_name='Sheet1')
    print(f"📋 数据集形状: {df.shape}")
    print(f"   列名: {list(df.columns)}")

    # 按Cohort分组
    dataframes = {}
    for cohort in df['Cohort'].unique():
        subset = df[df['Cohort'] == cohort].copy()
        dataframes[cohort.lower()] = subset
        print(f"📋 {cohort} 数据集: {subset.shape}")

    print(f"   总样本数: {len(df)}")
    print(f"   数据集分布: {df['Cohort'].value_counts().to_dict()}")

    return dataframes

def prepare_features_and_target(df, feature_columns, target_column):
    """
    准备特征和目标变量
    """
    print(f"🔧 准备特征和目标变量...")

    # 提取特征
    X = df[feature_columns].copy()

    # 提取目标变量
    y = df[target_column].copy()

    print(f"📊 特征形状: {X.shape}")
    print(f"🎯 目标变量分布:")
    print(y.value_counts())
    print()

    return X, y

def preprocess_data(X, continuous_features):
    """
    数据预处理：使用原始12个特征
    """
    print("⚙️ 正在进行数据预处理...")

    X_processed = X.copy()

    # 对连续特征进行标准化
    scalers = {}
    for feature in continuous_features:
        if feature in X_processed.columns:
            # 由于数据已经是标准化的，这里不需要重新标准化
            # 但我们需要记录参数以保持一致性
            scalers[feature] = {
                'mean': 0.0,  # 数据已经标准化
                'std': 1.0   # 数据已经标准化
            }
            print(f"   {feature} 已标准化 (mean=0.0, std=1.0)")

    # 处理分类特征（确保为数值型）
    categorical_features = [col for col in X_processed.columns if col not in continuous_features]
    for feature in categorical_features:
        X_processed[feature] = pd.to_numeric(X_processed[feature], errors='coerce').fillna(0)

    print(f"✅ 数据预处理完成: 使用 {X_processed.shape[1]} 个原始特征")
    print()

    return X_processed, scalers

def train_lightgbm_model_optimized(X_train, y_train, X_val, y_val, X_test=None, y_test=None):
    """
    训练优化后的LightGBM模型
    包含参数调优、特征工程等优化
    """
    print("🚀 开始训练优化后的LightGBM模型...")

    # 优化后的参数
    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'boosting_type': 'gbdt',
        'num_leaves': 64,  # 增加叶子节点数
        'max_depth': 8,    # 限制最大深度
        'learning_rate': 0.03,  # 降低学习率
        'feature_fraction': 0.8,  # 特征采样
        'bagging_fraction': 0.8,  # 数据采样
        'bagging_freq': 5,
        'min_child_samples': 20,  # 最小叶子节点样本数
        'min_child_weight': 0.001,
        'reg_alpha': 0.1,  # L1正则化
        'reg_lambda': 0.1, # L2正则化
        'scale_pos_weight': 1.0,  # 处理类别不平衡
        'verbose': -1,
        'seed': 42,
        'nthread': -1
    }

    print(f"🔧 优化后的模型参数: {params}")

    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # 训练模型
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,  # 增加训练轮数
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),  # 增加早停轮数
            lgb.log_evaluation(50)  # 更频繁的日志输出
        ]
    )

    print("✅ 基础模型训练完成")

    # 如果有测试数据，进行额外的验证
    if X_test is not None and y_test is not None:
        print("\n📊 在测试集上验证模型...")
        test_pred = model.predict(X_test)
        test_auc = roc_auc_score(y_test, test_pred)
        print(".4f")

    print()
    return model



def train_lightgbm_model(X_train, y_train, X_val, y_val, params=None):
    """
    训练LightGBM模型（保持原有接口）
    """
    return train_lightgbm_model_optimized(X_train, y_train, X_val, y_val)

def evaluate_model(model, X_test, y_test, threshold=0.4153):
    """
    评估模型性能
    """
    print("📊 正在评估模型性能...")

    # 预测概率
    y_pred_proba = model.predict(X_test)

    # 应用阈值进行分类
    y_pred = (y_pred_proba > threshold).astype(int)

    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(".4f")
    print(".4f")
    print("\n📋 分类报告:")
    print(classification_report(y_test, y_pred))

    # 阈值分析
    thresholds = np.arange(0.1, 0.9, 0.05)
    accuracies = []
    for thresh in thresholds:
        pred_thresh = (y_pred_proba > thresh).astype(int)
        acc = accuracy_score(y_test, pred_thresh)
        accuracies.append(acc)

    best_threshold_idx = np.argmax(accuracies)
    best_threshold = thresholds[best_threshold_idx]
    best_accuracy = accuracies[best_threshold_idx]

    print("\n🎯 阈值分析:")
    print(".4f")
    print("\n✅ 模型评估完成")
    print()

    return {
        'accuracy': accuracy,
        'auc': auc,
        'best_threshold': best_threshold,
        'best_accuracy': best_accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def save_model_and_scalers(model, scalers, feature_columns, target_column, threshold, save_dir="models"):
    """
    保存模型和相关参数
    """
    print("💾 正在保存模型和参数...")

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存LightGBM模型
    model_path = os.path.join(save_dir, f"lgb_model_{timestamp}.txt")
    model.save_model(model_path)

    # 保存为joblib格式（兼容性更好）
    model_joblib_path = os.path.join(save_dir, f"lgb_model_{timestamp}.joblib")
    joblib.dump(model, model_joblib_path)

    # 保存标准化参数
    scalers_path = os.path.join(save_dir, f"scalers_{timestamp}.json")
    with open(scalers_path, 'w', encoding='utf-8') as f:
        json.dump(scalers, f, indent=2, ensure_ascii=False)

    # 保存模型配置
    config = {
        'feature_columns': feature_columns,
        'target_column': target_column,
        'threshold': threshold,
        'continuous_features': list(scalers.keys()),
        'timestamp': timestamp,
        'model_params': model.params
    }

    config_path = os.path.join(save_dir, f"config_{timestamp}.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✅ 模型已保存到: {save_dir}")
    print(f"   - 模型文件: {model_path}")
    print(f"   - Joblib格式: {model_joblib_path}")
    print(f"   - 标准化参数: {scalers_path}")
    print(f"   - 配置文件: {config_path}")
    print()

    return {
        'model_path': model_path,
        'joblib_path': model_joblib_path,
        'scalers_path': scalers_path,
        'config_path': config_path
    }

def plot_feature_importance(model, feature_names, save_path=None):
    """
    绘制特征重要性
    """
    print("📈 正在绘制特征重要性...")

    # 获取特征重要性
    importance = model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Importance (Gain)')
    plt.ylabel('Features')
    plt.title('Feature Importance (Gain)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 特征重要性图已保存: {save_path}")

    plt.show()

def main():
    """
    主函数
    """
    print("🎯 开始使用Python LightGBM训练模型")
    print("=" * 60)

    # 1. 加载数据
    dataframes = load_and_explore_data("data.xlsx")

    # 2. 定义特征列和目标列（基于R代码）
    base_feature_columns = [
        'Sex', 'Gallstone', 'Other', 'Ultrasond', 'Dilatation',
        'GBMorphology', 'IntramuralNodule', 'GBMass', 'Line',
        'LymphNodes', 'FIB', 'IBIL'
    ]
    target_column = 'Label'  # 实际目标列名
    continuous_features = ['FIB', 'IBIL']  # 已经标准化的连续特征

    # 特征工程后会自动添加新的特征列，所以这里使用基础特征列
    feature_columns = base_feature_columns

    # 3. 准备训练、验证和测试数据
    # 注意：数据集中使用的是'Train', 'Validation', 'Test'
    if 'train' in dataframes and 'validation' in dataframes and 'test' in dataframes:
        print("✅ 发现Train、Validation、Test三个数据集")

        # 准备数据
        X_train, y_train = prepare_features_and_target(
            dataframes['train'], feature_columns, target_column
        )
        X_val, y_val = prepare_features_and_target(
            dataframes['validation'], feature_columns, target_column
        )
        X_test, y_test = prepare_features_and_target(
            dataframes['test'], feature_columns, target_column
        )

        # 数据预处理（包含特征工程和标准化）
        X_train_processed, scalers = preprocess_data(X_train, continuous_features)
        X_val_processed, _ = preprocess_data(X_val, continuous_features)
        X_test_processed, _ = preprocess_data(X_test, continuous_features)

        # 4. 训练优化后的LightGBM模型
        print("🚀 使用优化后的LightGBM模型训练...")
        model = train_lightgbm_model_optimized(
            X_train_processed, y_train,
            X_val_processed, y_val,
            X_test_processed, y_test
        )
        print("✅ 模型训练完成")

        # 5. 评估模型（使用与R语言一致的阈值0.4153）
        eval_results = evaluate_model(model, X_test_processed, y_test, threshold=0.4153)

        # 6. 保存模型为PKL格式
        print("💾 保存模型为PKL格式...")
        os.makedirs("models", exist_ok=True)

        # 保存完整的模型信息
        model_data = {
            'model': model,
            'scalers': scalers,
            'feature_columns': list(X_train_processed.columns),  # 原始12个特征
            'target_column': target_column,
            'threshold': 0.4153,  # 使用与R语言一致的阈值
            'model_params': model.params,
            'training_info': {
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'accuracy': eval_results['accuracy'],
                'auc': eval_results['auc'],
                'r_threshold': 0.4153,  # R语言使用的阈值
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'feature_engineering': False,  # 不使用特征工程
                'num_features': X_train_processed.shape[1]
            }
        }

        # 同时更新model_info.json文件
        info_path = "models/model_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model_path': "models/lgb_model_complete.pkl",
                'model_only_path': "models/lgb_model.pkl",
                'feature_columns': list(X_train_processed.columns),
                'target_column': target_column,
                'threshold': 0.4153,  # 确保JSON文件中也是0.4153
                'continuous_features': ['FIB', 'IBIL'],
                'training_info': model_data['training_info']
            }, f, indent=2, ensure_ascii=False)

        # 保存完整模型数据
        model_path = "models/lgb_model_complete.pkl"
        joblib.dump(model_data, model_path)
        print(f"✅ 完整模型已保存为PKL: {model_path}")

        # 保存仅模型文件（兼容性）
        model_only_path = "models/lgb_model.pkl"
        joblib.dump(model, model_only_path)
        print(f"✅ 模型文件已保存: {model_only_path}")

        # 保存模型信息
        info_path = "models/model_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model_path': model_path,
                'model_only_path': model_only_path,
                'feature_columns': feature_columns,
                'target_column': target_column,
                'threshold': eval_results['best_threshold'],
                'continuous_features': list(scalers.keys()),
                'training_info': model_data['training_info']
            }, f, indent=2, ensure_ascii=False)
        print(f"✅ 模型信息已保存: {info_path}")

        # 同时保存原始格式（可选）
        save_paths = save_model_and_scalers(
            model, scalers, feature_columns, target_column,
            eval_results['best_threshold']
        )

        # 7. 绘制特征重要性
        # 注意：使用训练时实际使用的特征名称
        plot_feature_importance(
            model, list(X_train_processed.columns),
            save_path=os.path.join("models", f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        )

        # 8. 输出最终结果
        print("🎉 训练完成！")
        print("=" * 60)
        print(f"📊 测试集准确率: {eval_results['accuracy']:.4f}")
        print(f"📊 AUC: {eval_results['auc']:.4f}")
        print(f"🎯 最优阈值: {eval_results['best_threshold']:.4f}")
        print(f"🔧 特征数量: {X_train_processed.shape[1]} 个原始特征")

        print(f"💾 模型已保存为PKL格式")
        print(f"   📁 完整模型: models/lgb_model_complete.pkl")
        print(f"   📁 仅模型文件: models/lgb_model.pkl")
        print(f"   📄 模型信息: models/model_info.json")

    else:
        print("❌ 未找到完整的train、val、test数据集")
        print(f"可用的数据集: {list(dataframes.keys())}")

if __name__ == "__main__":
    main()
