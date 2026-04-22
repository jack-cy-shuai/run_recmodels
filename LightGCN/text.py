import sys
import torch
import recbole

print("✅ 当前使用的Python解释器路径：", sys.executable)
print("✅ PyTorch版本：", torch.__version__)
print("✅ RecBole版本：", recbole.__version__)
print("✅ CUDA是否可用：", torch.cuda.is_available())