import numpy as np

def hit_rate_at_k(predictions, labels, k=10):
    """计算HR@k"""
    hits = 0
    for pred, label in zip(predictions, labels):
        # 按预测值排序，取前k个
        top_k_indices = np.argsort(pred)[-k:]
        if 1 in label[top_k_indices]:
            hits += 1
    return hits / len(predictions)

def ndcg_at_k(predictions, labels, k=10):
    """计算NDCG@k"""
    ndcg = 0
    for pred, label in zip(predictions, labels):
        # 按预测值排序
        sorted_indices = np.argsort(pred)[::-1]
        # 计算DCG
        dcg = 0
        for i in range(min(k, len(sorted_indices))):
            if label[sorted_indices[i]] == 1:
                dcg += 1 / np.log2(i + 2)
        # 计算IDCG
        ideal_sorted_indices = np.argsort(label)[::-1]
        idcg = 0
        for i in range(min(k, len(ideal_sorted_indices))):
            if label[ideal_sorted_indices[i]] == 1:
                idcg += 1 / np.log2(i + 2)
        if idcg > 0:
            ndcg += dcg / idcg
    return ndcg / len(predictions)

def evaluate_model(model, test_data, num_items, k=10):
    """评估模型性能"""
    # 为每个用户准备测试数据
    user_ids = test_data['user_id'].unique()
    predictions = []
    labels = []
    
    for user_id in user_ids:
        # 获取用户的正样本
        positive_items = test_data[(test_data['user_id'] == user_id) & (test_data['label'] == 1)]['item_id'].values
        # 随机选择99个负样本
        all_items = set(range(num_items))
        negative_items = list(all_items - set(positive_items))
        negative_items = np.random.choice(negative_items, 99, replace=False)
        # 合并正样本和负样本
        test_items = np.concatenate([positive_items, negative_items])
        # 生成用户输入
        user_input = np.full(len(test_items), user_id)
        # 预测 - 移除batch_size和verbose参数
        pred = model.predict(user_input, test_items).flatten()
        # 生成标签
        label = np.zeros(len(test_items))
        label[0] = 1  # 第一个是正样本
        # 添加到列表
        predictions.append(pred)
        labels.append(label)
    
    # 计算指标
    hr = hit_rate_at_k(predictions, labels, k)
    ndcg = ndcg_at_k(predictions, labels, k)
    return hr, ndcg