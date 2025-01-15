#信用评分数据处理
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('Credit_score_classification.csv')

# 转换特定列的数据类型（如数值列）
numeric_columns = ["Age", "Annual_Income", "Monthly_Inhand_Salary", 
    "Outstanding_Debt", "Credit_Utilization_Ratio"]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 标记空值
df.replace({"_": np.nan, "NA": np.nan, "NM": np.nan}, inplace=True)

# 删除含空值的行
df.dropna(inplace=True)

# 删除不合理的异常值
df = df[df["Age"] > 0]  # 年龄必须为正
df = df[df["Annual_Income"] > 0]  # 年收入必须为正
df = df[df["Credit_Utilization_Ratio"].between(0, 100)]  # 信用利用率在0-100之间


#打印需要标签编码的特征所包含的值
# for column in ["Occupation","Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour", "Credit_Score"]:
#     print(f"{column}: {set(df[column])}")

"""
Occupation 和 Payment_Behaviour：无特定顺序，按默认字母升序编码。
Credit_Mix 和 Credit_Score：按类别的业务顺序 (Bad < Standard < Good，Poor < Standard < Good) 编码。
Payment_of_Min_Amount：按布尔逻辑 (No = 0, Yes = 1) 编码。
"""
# 自定义排序规则
credit_mix_order = {"Bad": 0, "Standard": 1, "Good": 2}
credit_score_order = {"Poor": 0, "Standard": 1, "Good": 2}
payment_min_order = {"No": 0, "Yes": 1}

# 编码其他列
df["Occupation"] = df["Occupation"].astype("category").cat.codes  # 按字母顺序编码
df["Credit_Mix"] = df["Credit_Mix"].map(credit_mix_order)  # 按业务顺序编码
df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].map(payment_min_order)  # 布尔逻辑
df["Payment_Behaviour"] = df["Payment_Behaviour"].astype("category").cat.codes  # 按字母顺序编码
df["Credit_Score"] = df["Credit_Score"].map(credit_score_order)  # 按业务顺序编码

# 处理 Type_of_Loan 多标签编码
unique_loans = set(
    loan.strip() for loans in df["Type_of_Loan"].dropna() for loan in loans.split(",")
)
unique_loans = sorted(unique_loans)  # 按字母顺序排序
loan_mapping = {loan: idx for idx, loan in enumerate(unique_loans)}

# 创建多标签编码列
for loan in unique_loans:
    df[f"Loan_{loan.replace(' ', '_')}"] = df["Type_of_Loan"].apply(
        lambda x: 1 if loan in str(x) else 0
    )

# 删除原 Type_of_Loan 列
df.drop("Type_of_Loan", axis=1, inplace=True)

# 初始化Min-Max标准化器
scaler = MinMaxScaler()

# 对指定列进行标准化
columns_to_scale = [
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Credit_History_Age",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
]

df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])


# 输出预处理后的数据
print(df.head())

df.to_csv('clean_Credit_score_classification.csv', index=False)









