import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans, Birch, MiniBatchKMeans
from sklearn.cluster import BisectingKMeans
import dataframe_image as dfi

# 假设数据已经加载到DataFrame中
df = pd.read_csv("clean_index_transformed_Credit_score_classification.csv")  # 替换为实际路径

# 1. 数据预处理
#特征选择
features = [
    "Age",
    "Occupation",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Payment_of_Min_Amount",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Payment_Behaviour",
    "Monthly_Balance",
    "Credit_Score",
    "Credit_Mix_PCA_Component_1",
]

X = df[[features]]
X = df.dropna()  # 删除缺失值，或使用其他方式填充缺失值
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 定义评估函数
def evaluate_clustering(model, data, model_name):
    labels = model.fit_predict(data)
    silhouette_avg = silhouette_score(data, labels)
    ch_index = calinski_harabasz_score(data, labels)
    return {
        "Model": model_name,
        "Silhouette Coefficient": silhouette_avg,
        "Calinski-Harabasz Index": ch_index
    }

# 3. 初始化模型并评估
results = []

# KMeans
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
results.append(evaluate_clustering(kmeans, X_scaled, "KMeans"))

# BIRCH
birch = Birch(n_clusters=3)
results.append(evaluate_clustering(birch, X_scaled, "BIRCH"))

# Bisecting KMeans
bisecting_kmeans = BisectingKMeans(n_clusters=3, random_state=42)
results.append(evaluate_clustering(bisecting_kmeans, X_scaled, "Bisecting KMeans"))

# MiniBatchKMeans
minibatch_kmeans = MiniBatchKMeans(n_clusters=3, random_state=42)
results.append(evaluate_clustering(minibatch_kmeans, X_scaled, "MiniBatch KMeans"))

# 4. 将结果存储为表格
results_df = pd.DataFrame(results)
dfi.export(results_df, "信用评分数据聚类评估结果.png")  # 导出为图片
print(results_df)