import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from google.colab import files
# uploaded = files.upload()

data_df = pd.read_csv("Data.csv")
data_df.head()

data_df.info()
for col in data_df.columns :
  missing_data = data_df[col].isna().sum()
  missing_percent = missing_data / len(data_df) * 100
  print(f"Columns : {col} has {missing_percent} % missing data")

fig,ax = plt.subplots(figsize=(8,5))
sns.heatmap(data_df.isna(),cmap = "Blues", cbar= False, yticklabels=False)

#x = data_df.iloc[:,:-1] # tra ve dang dataframe
x = data_df.iloc[:,:-1].values # tra ve dang ndarray
y = data_df.iloc[:,-1].values
x,

from sklearn.impute import SimpleImputer
# tạo 1 biến instance để thay thế giá trị nan
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
x

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers = [("encoder",OneHotEncoder(),[0])],remainder = "passthrough")
x = ct.fit_transform(x)
x

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
y

from sklearn.model_selection import train_test_split
np.random.seed(42)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train[:,3:] = sc.fit_transform(x_train[:,3:])
x_test[:,3:] = sc.transform(x_test[:,3:])

x_train,x_test

