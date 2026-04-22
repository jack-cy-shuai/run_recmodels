import torch
import torch.nn as nn
import numpy as np

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # 初始化用户和物品的嵌入
        self.embedding = nn.Embedding(num_users + num_items, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        
    def forward(self, adj_matrix):
        # 获取初始嵌入
        x = self.embedding.weight
        
        # 存储每一层的嵌入
        emb_list = [x]
        
        # 多层图卷积
        for _ in range(self.num_layers):
            # 邻居聚合操作
            x = torch.sparse.mm(adj_matrix, x)
            emb_list.append(x)
        
        # 对各层嵌入进行平均
        output = torch.mean(torch.stack(emb_list), dim=0)
        
        return output
    
    def get_embeddings(self, adj_matrix):
        # 获取所有用户和物品的嵌入
        return self.forward(adj_matrix)
    
    def predict(self, user_ids, item_ids, adj_matrix):
        # 预测用户对物品的评分
        # 只执行一次前向传播
        all_emb = self.forward(adj_matrix)
        
        # 获取用户嵌入
        user_emb = all_emb[user_ids]
        
        # 获取物品嵌入
        item_emb = all_emb[self.num_users + item_ids.long()]
        
        # 计算得分
        scores = torch.sum(user_emb * item_emb, dim=1)
        return scores
