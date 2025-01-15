import pandas as pd
from sklearn.cluster import KMeans, Birch, MiniBatchKMeans, BisectingKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import dataframe_image as dfi
# 读取数据
df = pd.read_csv("clean_XAU_15m_data_2004_to_2024-20-09.csv")

# 选择特征
features = ["High_Low_Diff", "Close_Open_Diff"]
data = df[features]

# 数据标准化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# 存储评估结果
evaluation_results = []

# 1. K-means 聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_data)
kmeans_silhouette = silhouette_score(scaled_data, kmeans_labels)
kmeans_ch_score = calinski_harabasz_score(scaled_data, kmeans_labels)
evaluation_results.append(["KMeans", kmeans_silhouette, kmeans_ch_score])

# 2. Bisecting K-means 聚类
bisect_kmeans = BisectingKMeans(n_clusters=3, random_state=42)
bisect_labels = bisect_kmeans.fit_predict(scaled_data)
bisect_silhouette = silhouette_score(scaled_data, bisect_labels)
bisect_ch_score = calinski_harabasz_score(scaled_data, bisect_labels)
evaluation_results.append(["BisectingKMeans", bisect_silhouette, bisect_ch_score])

# 3. BIRCH 聚类
birch = Birch(n_clusters=3)
birch_labels = birch.fit_predict(scaled_data)
birch_silhouette = silhouette_score(scaled_data, birch_labels)
birch_ch_score = calinski_harabasz_score(scaled_data, birch_labels)
evaluation_results.append(["BIRCH", birch_silhouette, birch_ch_score])

# 4. MiniBatch K-means 聚类
minibatch_kmeans = MiniBatchKMeans(n_clusters=3, random_state=42)
minibatch_labels = minibatch_kmeans.fit_predict(scaled_data)
minibatch_silhouette = silhouette_score(scaled_data, minibatch_labels)
minibatch_ch_score = calinski_harabasz_score(scaled_data, minibatch_labels)
evaluation_results.append(["MiniBatchKMeans", minibatch_silhouette, minibatch_ch_score])

# 将评估结果转换为 DataFrame 并输出
evaluation_df = pd.DataFrame(evaluation_results, columns=["Model", "Silhouette Coefficient", 
	"Calinski-Harabasz Index"])
dfi.export(evaluation_df, "黄金价格数据聚类评估结果.png")  # 导出为图片
print(evaluation_df)
