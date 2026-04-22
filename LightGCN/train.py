import torch
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from model import LightGCN
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
# 设置matplotlib后端为Agg，避免显示问题
matplotlib.use('Agg')

class DataProcessor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.num_users = None
        self.num_items = None
        self.train_data = None
        self.test_data = None
        self.adj_matrix = None
    
    def load_data(self):
        # 加载MovieLens-100K数据集
        if not os.path.exists(self.dataset_path):
            print(f"数据集路径不存在: {self.dataset_path}")
            return False
        
        # 读取评分数据
        df = pd.read_csv(self.dataset_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
        
        # 处理用户和物品ID，使其从0开始
        df['user_id'] = df['user_id'] - 1
        df['item_id'] = df['item_id'] - 1
        
        # 统计用户和物品数量
        self.num_users = df['user_id'].max() + 1
        self.num_items = df['item_id'].max() + 1
        
        print("数据集加载完成: %d用户, %d物品" % (self.num_users, self.num_items))
        return df
    
    def split_data(self, df, test_size=0.2):
        # 划分训练集和测试集
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        
        # 构建训练数据字典
        self.train_data = {}
       # 构建正样本对列表，用于批量处理
        self.train_pairs = []
        for _, row in train_df.iterrows():
            user_id = int(row['user_id'])
            item_id = int(row['item_id'])
            if user_id not in self.train_data:
                self.train_data[user_id] = []
            self.train_data[user_id].append(item_id)
            self.train_pairs.append((user_id, item_id))
        
        # 构建测试数据字典
        self.test_data = {}
        for _, row in test_df.iterrows():
            user_id = int(row['user_id'])
            item_id = int(row['item_id'])
            if user_id not in self.test_data:
                self.test_data[user_id] = []
            self.test_data[user_id].append(item_id)
        
        print("数据划分完成: 训练集%d条, 测试集%d条" % (len(train_df), len(test_df)))
    
    def build_adj_matrix(self):
        # 构建用户-物品交互矩阵
        total_nodes = self.num_users + self.num_items
        edges = []
        
        # 添加用户-物品边
        for user_id, items in self.train_data.items():
            for item_id in items:
                edges.append([user_id, self.num_users + item_id])
                edges.append([self.num_users + item_id, user_id])
        
        # 构建稀疏邻接矩阵
        edges = np.array(edges).T
        values = np.ones(edges.shape[1], dtype=np.float32)
        
        # 计算度矩阵
        deg = np.zeros(total_nodes, dtype=np.float32)
        for i in range(edges.shape[1]):
            deg[edges[0, i]] += 1
        
        # 计算归一化因子
        deg_inv_sqrt = np.zeros_like(deg)
        non_zero_indices = deg > 0
        deg_inv_sqrt[non_zero_indices] = np.power(deg[non_zero_indices], -0.5)
        
        # 构建归一化的邻接矩阵
        row = edges[0]
        col = edges[1]
        weight = values * deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # 创建稀疏矩阵
        indices = np.array([row, col])
        self.adj_matrix = torch.sparse_coo_tensor(
            torch.tensor(indices),
            torch.tensor(weight),
            (total_nodes, total_nodes)
        )
        
        print("邻接矩阵构建完成: %s" % str(self.adj_matrix.shape))

class Trainer:
    def __init__(self, model, data_processor, learning_rate=0.001, batch_size=256, num_epochs=100):
        self.model = model
        self.data_processor = data_processor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 用于记录训练过程
        self.history = {
            'loss': [],
            'recall': [],
            'precision': [],
            'ndcg': []
        }
    
    def sample_negative_batch(self, user_ids, num_neg=1):
        # 批量负采样
        neg_items = []
        for user_id in user_ids:
            all_items = set(range(self.data_processor.num_items))
            pos_items = set(self.data_processor.train_data.get(user_id, []))
            neg_candidates = list(all_items - pos_items)
            if neg_candidates:
                neg_item = np.random.choice(neg_candidates, size=num_neg, replace=False)
                neg_items.append(neg_item[0])
            else:
                # 如果没有负样本，随机选择一个物品
                neg_items.append(np.random.randint(0, self.data_processor.num_items))
        return neg_items
    
    def train_epoch(self):
        # 训练一个epoch
        self.model.train()
        total_loss = 0
        
        # 打乱训练数据
        np.random.shuffle(self.data_processor.train_pairs)
        
        # 批量处理
        batch_size = self.batch_size
        num_batches = len(self.data_processor.train_pairs) // batch_size + 1
        
        # 使用tqdm创建进度条
        with tqdm(total=num_batches, desc="训练中", unit="批次") as pbar:
            for batch_idx in range(num_batches):
                # 获取当前批次的正样本对
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, len(self.data_processor.train_pairs))
                batch_pairs = self.data_processor.train_pairs[start:end]
                
                if not batch_pairs:
                    continue
                
                # 提取用户ID和物品ID
                user_ids = [pair[0] for pair in batch_pairs]
                pos_items = [pair[1] for pair in batch_pairs]
                
                # 批量负采样
                neg_items = self.sample_negative_batch(user_ids)
                
                # 转换为张量并移至设备
                user_ids_tensor = torch.tensor(user_ids, device=self.device)
                pos_items_tensor = torch.tensor(pos_items, device=self.device)
                neg_items_tensor = torch.tensor(neg_items, device=self.device)
                adj_matrix = self.data_processor.adj_matrix.to(self.device)
                
                # 计算正样本和负样本的分数
                pos_scores = self.model.predict(user_ids_tensor, pos_items_tensor, adj_matrix)
                neg_scores = self.model.predict(user_ids_tensor, neg_items_tensor, adj_matrix)
                
                # 计算损失
                loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).sum()
                total_loss += loss.item()
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({'当前损失': loss.item() / len(batch_pairs)})
        
        return total_loss
    
    def evaluate(self, k=10):
        # 评估模型
        self.model.eval()
        recall = 0
        precision = 0
        ndcg = 0
        user_count = 0
        
        # 使用tqdm创建评估进度条
        with tqdm(total=len(self.data_processor.test_data), desc="评估中", unit="用户") as pbar:
            with torch.no_grad():
                # 预计算邻接矩阵
                adj_matrix = self.data_processor.adj_matrix.to(self.device)
                
                for user_id in self.data_processor.test_data:
                    # 获取测试集中的正样本
                    test_items = self.data_processor.test_data[user_id]
                    if not test_items:
                        pbar.update(1)
                        continue
                    
                    # 获取用户已交互的物品
                    interacted_items = set(self.data_processor.train_data.get(user_id, []))
                    
                    # 生成候选物品（排除已交互的）
                    candidate_items = [i for i in range(self.data_processor.num_items) if i not in interacted_items]
                    
                    if not candidate_items:
                        pbar.update(1)
                        continue
                    
                    # 预测分数
                    user_ids = torch.tensor([user_id] * len(candidate_items), device=self.device)
                    item_ids = torch.tensor(candidate_items, device=self.device)
                    scores = self.model.predict(user_ids, item_ids, adj_matrix)
                    
                    # 排序并获取top-k
                    top_k_indices = torch.argsort(scores, descending=True)[:k]
                    top_k_items = [candidate_items[i] for i in top_k_indices.cpu().numpy()]
                    
                    # 计算评估指标
                    true_positives = len(set(top_k_items) & set(test_items))
                    recall += true_positives / len(test_items)
                    precision += true_positives / k
                    
                    # 计算NDCG
                    dcg = 0
                    idcg = 0
                    for i, item in enumerate(top_k_items):
                        if item in test_items:
                            dcg += 1 / np.log2(i + 2)
                    for i in range(min(len(test_items), k)):
                        idcg += 1 / np.log2(i + 2)
                    ndcg += dcg / idcg if idcg > 0 else 0
                    
                    user_count += 1
                    pbar.update(1)
        
        if user_count == 0:
            return 0, 0, 0
        
        return recall / user_count, precision / user_count, ndcg / user_count
    
    def plot_training_history(self):
        # 绘制训练历史
        plt.figure(figsize=(15, 10))
        
        # 绘制损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(self.history['loss'], label='Loss')
        plt.title('训练损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # 绘制Recall曲线
        plt.subplot(2, 2, 2)
        plt.plot(self.history['recall'], label='Recall@10')
        plt.title('Recall@10')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.grid(True)
        plt.legend()
        
        # 绘制Precision曲线
        plt.subplot(2, 2, 3)
        plt.plot(self.history['precision'], label='Precision@10')
        plt.title('Precision@10')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.grid(True)
        plt.legend()
        
        # 绘制NDCG曲线
        plt.subplot(2, 2, 4)
        plt.plot(self.history['ndcg'], label='NDCG@10')
        plt.title('NDCG@10')
        plt.xlabel('Epoch')
        plt.ylabel('NDCG')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("训练历史图表已保存为 training_history.png")
    
    def train(self):
        # 训练模型
        print("开始训练，使用设备: %s" % str(self.device))
        print("=" * 80)
        
        for epoch in range(self.num_epochs):
            print("\nEpoch %d/%d" % (epoch+1, self.num_epochs))
            print("-" * 80)
            
            # 训练
            loss = self.train_epoch()
            
            # 评估
            recall, precision, ndcg = self.evaluate()
            
            # 记录历史数据
            self.history['loss'].append(loss)
            self.history['recall'].append(recall)
            self.history['precision'].append(precision)
            self.history['ndcg'].append(ndcg)
            
            # 输出结果
            print("-" * 80)
            print("Epoch %d 结果:" % (epoch+1))
            print("Loss: %.4f" % loss)
            print("Recall@10: %.4f" % recall)
            print("Precision@10: %.4f" % precision)
            print("NDCG@10: %.4f" % ndcg)
            print("=" * 80)
        
        # 绘制训练历史
        self.plot_training_history()
        
        # 最终评估
        final_recall, final_precision, final_ndcg = self.evaluate()
        print("\n最终评估结果:")
        print("-" * 80)
        print("Recall@10: %.4f" % final_recall)
        print("Precision@10: %.4f" % final_precision)
        print("NDCG@10: %.4f" % final_ndcg)
        print("-" * 80)

if __name__ == "__main__":
    # 配置
    dataset_path = "./ml-100k/u.data"
    embedding_dim = 64
    num_layers = 3
    learning_rate = 0.001
    batch_size = 64  # 减小批次大小以减少内存使用
    num_epochs = 5
    
    # 数据处理
    data_processor = DataProcessor(dataset_path)
    df = data_processor.load_data()
    if df is not None:
        data_processor.split_data(df)
        data_processor.build_adj_matrix()
        
        # 构建模型
        model = LightGCN(
            num_users=data_processor.num_users,
            num_items=data_processor.num_items,
            embedding_dim=embedding_dim,
            num_layers=num_layers
        )
        
        # 训练
        trainer = Trainer(
            model=model,
            data_processor=data_processor,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs
        )
        trainer.train()
