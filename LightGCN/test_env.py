import torch
import numpy as np
import pandas as pd

print("测试环境配置")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print(f"NumPy版本: {np.__version__}")
print(f"Pandas版本: {pd.__version__}")

# 测试基本的PyTorch操作
print("\n测试PyTorch基本操作")
test_tensor = torch.tensor([1, 2, 3])
print(f"测试张量: {test_tensor}")

# 测试CUDA（如果可用）
if torch.cuda.is_available():
    test_tensor_cuda = test_tensor.to('cuda')
    print(f"CUDA张量: {test_tensor_cuda}")

print("\n环境测试完成")
