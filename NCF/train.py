import numpy as np
import pandas as pd
import os
from data_preprocess import prepare_data
from ncf_model import create_ncf
from evaluation import evaluate_model

# 设置随机种子
np.random.seed(42)

# 准备数据
train_samples, val_samples, test_samples, num_users, num_items = prepare_data()

# 提取特征和标签
train_user = train_samples['user_id'].values
train_item = train_samples['item_id'].values
train_label = train_samples['label'].values

val_user = val_samples['user_id'].values
val_item = val_samples['item_id'].values
val_label = val_samples['label'].values

# 创建模型
model = create_ncf(num_users, num_items, mf_dim=8, mlp_dim=[64, 32, 16, 8], l2_reg=0.001)

# 训练模型
model.fit(train_user, train_item, train_label, epochs=50, batch_size=256)

# 评估模型
hr, ndcg = evaluate_model(model, test_samples, num_items, k=10)
print(f"HR@10: {hr:.4f}")
print(f"NDCG@10: {ndcg:.4f}")

# 保存训练日志
with open('training_log.txt', 'w') as f:
    f.write('Training Log\n')
    f.write('\nEvaluation Results\n')
    f.write(f"HR@10: {hr:.4f}\n")
    f.write(f"NDCG@10: {ndcg:.4f}\n")