import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

df = pd.read_csv('clean_Credit_score_classification.csv')

features = [
    "Age", "Occupation", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", 
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", 
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries", 
    "Credit_Mix", "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age", 
    "Payment_of_Min_Amount", "Total_EMI_per_month", "Amount_invested_monthly", 
    "Payment_Behaviour", "Monthly_Balance", "Credit_Score"
]

df=df[features]
# 计算所有特征与目标变量的相关性
correlation_matrix = df.corr()
credit_score_corr = correlation_matrix["Credit_Score"].sort_values(ascending=False)

# 设定相关性阈值，筛选与目标变量相关性较高的特征
threshold = 0.2
high_corr_features = credit_score_corr[abs(credit_score_corr) > threshold].index.tolist()

# 显示与 Credit_Score 相关性较高的特征
print("与 Credit_Score 相关性较高的特征:")
print(credit_score_corr[high_corr_features])

# 计算相关性矩阵并绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix.loc[high_corr_features, high_corr_features], 
    annot=True, fmt=".2f", cmap="coolwarm")
plt.title("特征相关性热力图")
plt.show()

# 根据领域知识和相关性热力图，删除高度相关的特征，保留重要特征
# 示例：删除高度相关的特征，避免重复信息
selected_features = [
    "Credit_Mix","Credit_History_Age","Annual_Income","Monthly_Inhand_Salary"
]  # 这里根据领域知识和相关性热力图手动选择


# 提取用于 PCA 的数据
pca_data = df[selected_features]

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 对 selected_features 数据进行标准化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pca_data)

# 执行 PCA 分析
pca = PCA(n_components=0.75)  # 保留解释75%波动率的主成分
pca_transformed = pca.fit_transform(scaled_data)

# 输出主成分信息
print(f"主成分数目: {pca.n_components_}")
print("各主成分解释的方差比率:")
print(pca.explained_variance_ratio_)
print(f"累计解释的方差比率: {sum(pca.explained_variance_ratio_):.2f}")

# 将主成分添加到原数据框 df 中
for i in range(pca.n_components_):
    df[f"Credit_Mix_PCA_Component_{i+1}"] = pca_transformed[:, i]

# 从 df 中删除原始特征
df.drop(columns=selected_features, inplace=True)

# 显示处理后的数据框
print("添加主成分后的数据框:")
print(df.head())

# # 保存处理后的数据框到新的 CSV 文件
# df.to_csv("pca_transformed_data.csv", index=False)
# print("主成分分析完成，并已保存为 pca_transformed_data.csv")



#2.信息增益法针对贷款类型进行分析（只适合离散变量）
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


data_frame = pd.read_csv("clean_Credit_score_classification.csv")  
# 将目标变量 Credit_Score 转换为数值型
le = LabelEncoder()
data_frame['Credit_Score'] = le.fit_transform(data_frame['Credit_Score'])

# 定义自变量
independent_vars = [
    'Loan_Auto_Loan', 'Loan_Credit-Builder_Loan', 'Loan_Debt_Consolidation_Loan', 
    'Loan_Home_Equity_Loan', 'Loan_Mortgage_Loan', 'Loan_No_Data', 'Loan_Not_Specified', 
    'Loan_Payday_Loan', 'Loan_Personal_Loan', 'Loan_Student_Loan', 
    'Loan_and_Auto_Loan', 'Loan_and_Credit-Builder_Loan', 'Loan_and_Debt_Consolidation_Loan', 
    'Loan_and_Home_Equity_Loan', 'Loan_and_Mortgage_Loan', 'Loan_and_Not_Specified', 
    'Loan_and_Payday_Loan', 'Loan_and_Personal_Loan', 'Loan_and_Student_Loan'
]

# 获取自变量 X 和因变量 y
X = data_frame[independent_vars]
y = data_frame['Credit_Score']

# 使用决策树模型计算特征重要性（信息增益）
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

# 获取每个特征的信息增益（重要性）
feature_importance = clf.feature_importances_

# 将特征名称和对应的重要性值放入 DataFrame 中
importance_df = pd.DataFrame({
    'Feature': independent_vars,
    'Information Gain': feature_importance
})

# 按照信息增益排序，并选择前五个特征
top_5_features = importance_df.sort_values(by='Information Gain', ascending=False).head(5)

print('信息增益前五的特征：\n',top_5_features)
# 获取前五个特征名称
top_5_feature_names = top_5_features['Feature'].tolist()

df[top_5_feature_names]=data_frame[top_5_feature_names]

df.to_csv("index_transformed_data.csv", index=False)
print("指标处理完成，并已保存为 index_transformed_data.csv")

