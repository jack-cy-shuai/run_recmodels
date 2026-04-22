import torch
import torch.nn as nn
import torch.optim as optim

class NCF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim=8, mlp_dim=[64, 32, 16, 8], dropout=0.2, l2_reg=0.001):
        super(NCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.mf_dim = mf_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.l2_reg = l2_reg
        
        # GMF嵌入层
        self.user_embedding_mf = nn.Embedding(num_users, mf_dim)
        self.item_embedding_mf = nn.Embedding(num_items, mf_dim)
        
        # MLP嵌入层
        self.user_embedding_mlp = nn.Embedding(num_users, mlp_dim[0]//2)
        self.item_embedding_mlp = nn.Embedding(num_items, mlp_dim[0]//2)
        
        # MLP层
        self.mlp_layers = nn.ModuleList()
        for i in range(len(mlp_dim) - 1):
            self.mlp_layers.append(nn.Linear(mlp_dim[i], mlp_dim[i+1]))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout))
        
        # 融合层
        self.fusion_layer = nn.Linear(mf_dim + mlp_dim[-1], 1)
        self.sigmoid = nn.Sigmoid()
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        # 初始化嵌入层
        nn.init.normal_(self.user_embedding_mf.weight, std=0.1)
        nn.init.normal_(self.item_embedding_mf.weight, std=0.1)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.1)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.1)
        
        # 初始化MLP层
        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.1)
                nn.init.zeros_(layer.bias)
        
        # 初始化融合层
        nn.init.normal_(self.fusion_layer.weight, std=0.1)
        nn.init.zeros_(self.fusion_layer.bias)
    
    def forward(self, user_ids, item_ids):
        # GMF部分
        user_embed_mf = self.user_embedding_mf(user_ids)
        item_embed_mf = self.item_embedding_mf(item_ids)
        mf_output = user_embed_mf * item_embed_mf
        
        # MLP部分
        user_embed_mlp = self.user_embedding_mlp(user_ids)
        item_embed_mlp = self.item_embedding_mlp(item_ids)
        mlp_output = torch.cat([user_embed_mlp, item_embed_mlp], dim=1)
        
        # MLP隐藏层
        for layer in self.mlp_layers:
            mlp_output = layer(mlp_output)
        
        # 融合GMF和MLP输出
        combined_output = torch.cat([mf_output, mlp_output], dim=1)
        # 输出层
        prediction = self.sigmoid(self.fusion_layer(combined_output))
        return prediction
    
    def fit(self, user_ids, item_ids, labels, epochs=50, batch_size=256, learning_rate=0.001):
        # 转换为张量
        user_ids = torch.tensor(user_ids, dtype=torch.long)
        item_ids = torch.tensor(item_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
        
        # 移至GPU（如果可用）
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        labels = labels.to(device)
        
        # 定义优化器和损失函数
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=self.l2_reg)
        criterion = nn.BCELoss()
        
        # 训练
        num_samples = len(user_ids)
        for epoch in range(epochs):
            # 打乱数据
            indices = torch.randperm(num_samples)
            user_ids_shuffled = user_ids[indices]
            item_ids_shuffled = item_ids[indices]
            labels_shuffled = labels[indices]
            
            # 批量训练
            total_loss = 0
            for i in range(0, num_samples, batch_size):
                batch_end = min(i + batch_size, num_samples)
                batch_user = user_ids_shuffled[i:batch_end]
                batch_item = item_ids_shuffled[i:batch_end]
                batch_label = labels_shuffled[i:batch_end]
                
                # 前向传播
                optimizer.zero_grad()
                pred = self(batch_user, batch_item)
                
                # 计算损失
                loss = criterion(pred, batch_label)
                total_loss += loss.item()
                
                # 反向传播
                loss.backward()
                optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/(num_samples//batch_size):.4f}")
    
    def predict(self, user_ids, item_ids):
        # 转换为张量
        user_ids = torch.tensor(user_ids, dtype=torch.long)
        item_ids = torch.tensor(item_ids, dtype=torch.long)
        
        # 移至GPU（如果可用）
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        
        # 预测
        with torch.no_grad():
            pred = self(user_ids, item_ids)
        
        return pred.cpu().numpy()

# 为了兼容原有代码，保留create_ncf函数
def create_ncf(num_users, num_items, mf_dim=8, mlp_dim=[64, 32, 16, 8], dropout=0.2, l2_reg=0.001):
    return NCF(num_users, num_items, mf_dim, mlp_dim, dropout, l2_reg)