import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
# 创建 DataFrame
df = pd.read_csv('diversified_ecommerce_dataset.csv')

# 初始化 LabelEncoder
label_encoder = LabelEncoder()

# 对 "Product Name", "Category", "Customer Location" 进行数值化处理
df["Product Name"] = label_encoder.fit_transform(df["Product Name"])
df["Category"] = label_encoder.fit_transform(df["Category"])
df["Customer Location"] = label_encoder.fit_transform(df["Customer Location"])

# 特定顺序编码: "Customer Age Group", "Shipping Method"
age_group_mapping = {"18-24": 1, "25-34": 2, "35-44": 3, "45-54": 4, "55+": 5}
shipping_method_mapping = {"Standard": 1, "Express": 2, "Overnight": 3}
df['Customer Age Group'] = df['Customer Age Group'].map(age_group_mapping)
df['Shipping Method'] = df['Shipping Method'].map(shipping_method_mapping)

# 二元变量编码: "Customer Gender", "Seasonality"
df['Customer Gender'] = df['Customer Gender'].map({"Male": 0, "Female": 1, "Non-Binary": 2})
df['Seasonality'] = df['Seasonality'].map({"No": 0, "Yes": 1})

# 离散化 Return Rate（选择等宽分箱或自定义分箱）
bins = [0, 5,15,25,100]  # 自定义分箱区间，表示退货率的不同区间
labels = [1, 2, 3,4]  # 分箱后的标签

# 使用 pd.cut() 进行离散化
df['Return Rate'] = pd.cut(df['Return Rate'], bins=bins, labels=labels, right=False)


# Find feature
features = ["Price", "Stock Level", "Shipping Cost", "Popularity Index"]

# Discretization function
def discretize(column, bins=5):
    return pd.cut(column, bins, labels=False)

# Apply discretization
discretized_df = df[features].apply(discretize)

# Replace original columns with discretized values
df[features] = discretized_df

#清洗空值
df = df.dropna()
df = df.drop('Supplier ID',axis=1)
df = df.drop('Product ID',axis=1)
df = df.drop('Product Name',axis=1)

# # 查看清洗和特征工程后的数据
# df.head()

df.to_csv("clean_diversified_ecommerce_dataset.csv", index=False)