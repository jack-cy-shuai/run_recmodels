import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_movielens_100k():
    """加载MovieLens-100K数据集"""
    # 数据集路径
    data_path = 'ml-100k/u.data'
    # 读取数据
    data = pd.read_csv(data_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    # 只保留评分>=4的作为正样本
    data = data[data['rating'] >= 4]
    # 重新编码用户和物品ID，从0开始
    user_ids = data['user_id'].unique()
    item_ids = data['item_id'].unique()
    user_id_map = {id: i for i, id in enumerate(user_ids)}
    item_id_map = {id: i for i, id in enumerate(item_ids)}
    data['user_id'] = data['user_id'].map(user_id_map)
    data['item_id'] = data['item_id'].map(item_id_map)
    # 获取用户和物品数量
    num_users = len(user_ids)
    num_items = len(item_ids)
    return data, num_users, num_items

def split_data(data, test_size=0.2, val_size=0.1):
    """划分训练集、验证集和测试集"""
    # 首先划分训练集和测试集
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    # 从训练集中划分出验证集
    train_data, val_data = train_test_split(train_data, test_size=val_size/(1-test_size), random_state=42)
    return train_data, val_data, test_data

def negative_sampling(data, num_items, negative_ratio=4):
    """负样本采样"""
    # 创建用户-物品交互字典
    user_item_dict = {}
    for _, row in data.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        if user_id not in user_item_dict:
            user_item_dict[user_id] = set()
        user_item_dict[user_id].add(item_id)
    
    # 生成负样本
    negative_samples = []
    for _, row in data.iterrows():
        user_id = row['user_id']
        # 采样负样本
        for _ in range(negative_ratio):
            while True:
                item_id = np.random.randint(0, num_items)
                if item_id not in user_item_dict[user_id]:
                    negative_samples.append({'user_id': user_id, 'item_id': item_id, 'label': 0})
                    break
    
    # 将正样本和负样本合并
    positive_samples = data[['user_id', 'item_id']].copy()
    positive_samples['label'] = 1
    all_samples = pd.concat([positive_samples, pd.DataFrame(negative_samples)], ignore_index=True)
    return all_samples

def prepare_data():
    """准备数据"""
    # 加载数据
    data, num_users, num_items = load_movielens_100k()
    # 划分数据集
    train_data, val_data, test_data = split_data(data)
    # 负样本采样
    train_samples = negative_sampling(train_data, num_items)
    val_samples = negative_sampling(val_data, num_items)
    test_samples = negative_sampling(test_data, num_items)
    return train_samples, val_samples, test_samples, num_users, num_items