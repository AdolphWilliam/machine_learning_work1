import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr

df = pd.read_csv('clean_Credit_score_classification.csv')


# 1.Pearson 相关系数
# 选择需要计算相关性的特征
features = [
    "Age", "Occupation", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", 
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", 
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries", 
    "Credit_Mix", "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age", 
    "Payment_of_Min_Amount", "Total_EMI_per_month", "Amount_invested_monthly", 
    "Payment_Behaviour", "Monthly_Balance", "Credit_Score"
]

# 确保选中的特征在数据中存在
df_selected = df[features]

#计算连续变量之间的Pearson 相关系数
correlation_matrix = df_selected.corr(method="pearson")

# 设置热力图的大小
plt.figure(figsize=(15, 12))

# 绘制热力图
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm", 
    cbar=True, 
    square=True, 
    annot_kws={"size": 8}
)

# 添加标题
plt.title("Feature Correlation Heatmap", fontsize=16)

# 显示图表
plt.tight_layout()
plt.show()


# 2.点二列相关系数（计算二元变量贷款类型与Credit_Score的相关性）
# 提取所需特征
loan_features = [
    "Loan_Auto_Loan", "Loan_Credit-Builder_Loan", "Loan_Debt_Consolidation_Loan", 
    "Loan_Home_Equity_Loan", "Loan_Mortgage_Loan", "Loan_No_Data", "Loan_Not_Specified", 
    "Loan_Payday_Loan", "Loan_Personal_Loan", "Loan_Student_Loan", "Loan_and_Auto_Loan", 
    "Loan_and_Credit-Builder_Loan", "Loan_and_Debt_Consolidation_Loan", 
    "Loan_and_Home_Equity_Loan", "Loan_and_Mortgage_Loan", "Loan_and_Not_Specified", 
    "Loan_and_Payday_Loan", "Loan_and_Personal_Loan", "Loan_and_Student_Loan"
]
target = "Credit_Score"

# 创建一个空的 DataFrame 用于存储相关系数
correlation_results = pd.DataFrame(columns=["Feature", "PointBiserial_Correlation"])

# 计算点二列相关系数
for feature in loan_features:
    if df[feature].notna().all() and df[target].notna().all():
        correlation, _ = pointbiserialr(df[feature], df[target])
        temp_df = pd.DataFrame({"Feature": [feature], "PointBiserial_Correlation": [correlation]})
        correlation_results = pd.concat([correlation_results, temp_df], ignore_index=True)

# 将相关系数按照特征名称排序
correlation_results.set_index("Feature", inplace=True)

# 设置热力图大小
plt.figure(figsize=(10, 6))

# 绘制热力图
sns.heatmap(
    correlation_results.T,  # 转置以在 x 轴显示特征
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm", 
    cbar=True, 
    annot_kws={"size": 10}
)

# 添加标题
plt.title("Point-Biserial Correlation Heatmap: Loans vs Credit_Score", fontsize=16)

# 显示图表
plt.tight_layout()
plt.show()