import os
import urllib.request
import zipfile

# 数据集下载链接
url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
# 保存路径
zip_path = "ml-100k.zip"

# 下载数据集
print("Downloading MovieLens-100K dataset...")
urllib.request.urlretrieve(url, zip_path)

# 解压数据集
print("Extracting dataset...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall()

# 删除zip文件
os.remove(zip_path)
print("Dataset downloaded and extracted successfully!")