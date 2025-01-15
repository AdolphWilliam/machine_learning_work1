import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans, Birch, MiniBatchKMeans, DBSCAN
import dataframe_image as dfi

# 数据加载
df = pd.read_csv("clean_diversified_ecommerce_dataset.csv")

# 选择产品特性列
product_features = ["Price", "Discount", "Tax Rate", "Stock Level", "Seasonality",
    "Popularity Index"]
X = df[product_features]

# 数据标准化
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
X_scaled = X

# 初始化评估结果表格
results = []

# 1. K-means 聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
kmeans_ch = calinski_harabasz_score(X_scaled, kmeans_labels)
results.append(["KMeans", kmeans_silhouette, kmeans_ch])
df['KMeans_Cluster'] = kmeans_labels

# 2. DBSCAN 聚类
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# 计算 DBSCAN 评价指标
if len(set(dbscan_labels)) > 1:  # 检查聚类结果是否合理
    dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels)
    dbscan_ch = calinski_harabasz_score(X_scaled, dbscan_labels)
else:  # 若只有一个簇或所有点均为噪声
    dbscan_silhouette = np.nan
    dbscan_ch = np.nan

results.append(["DBSCAN", dbscan_silhouette, dbscan_ch])
df['DBSCAN_Cluster'] = dbscan_labels

# 3. BIRCH 聚类
birch = Birch(n_clusters=3)
birch_labels = birch.fit_predict(X_scaled)
birch_silhouette = silhouette_score(X_scaled, birch_labels)
birch_ch = calinski_harabasz_score(X_scaled, birch_labels)
results.append(["BIRCH", birch_silhouette, birch_ch])
df['BIRCH_Cluster'] = birch_labels

# 4. MiniBatch K-means 聚类
minibatch_kmeans = MiniBatchKMeans(n_clusters=3, random_state=42)
minibatch_labels = minibatch_kmeans.fit_predict(X_scaled)
minibatch_silhouette = silhouette_score(X_scaled, minibatch_labels)
minibatch_ch = calinski_harabasz_score(X_scaled, minibatch_labels)
results.append(["MiniBatch KMeans", minibatch_silhouette, minibatch_ch])
df['MiniBatchKMeans_Cluster'] = minibatch_labels

# 输出评估结果
results_df = pd.DataFrame(results, columns=["Model", "Silhouette Coefficient", 
    "Calinski-Harabasz Index"])
dfi.export(results_df, "电子商务数据聚类评估结果.png")  # 导出为图片
print(results_df)
