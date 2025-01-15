import pandas as pd

df = pd.read_csv('index_transformed_data.csv')


# 1.计算变异系数并求权重
std = df.std()
mean = df.mean()
cv = std / mean
weights = cv / cv.sum()

# 计算新的特征值
df_weighted = df.drop(columns=["Credit_Score"]) * weights

# 导出为csv
output_path = "weighted_features.csv"
df_weighted.to_csv(output_path, index=False)

# 2.变异系数法效果较差，使用熵权法进行权重处理

from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 提取特征变量和目标变量
X = df.drop(columns=["Credit_Score"])
y = df["Credit_Score"]

# 训练随机森林模型
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y)

# 获取特征的重要性权重
importance = rf_model.feature_importances_
weights_rf = importance / importance.sum()

# 计算新的特征值
df_weighted_rf = X * weights_rf
df_weighted_rf["Credit_Score"]=df["Credit_Score"]
# 导出为csv
output_path_rf = "weighted_features_rf.csv"
df_weighted_rf.to_csv(output_path_rf, index=False)


