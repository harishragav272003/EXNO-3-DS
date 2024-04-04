## EXNO-3-DS

```
NAME : HARISH RAGAV S
REG NO : 212222110013
```

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
Done by-
Name- KISHORE KUMAR U
Reg No- 212222233003
import pandas as pd
df=pd.read_csv('/content/Encoding Data.csv')
df
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/03f8eed4-910f-4b36-952a-e42f04af61d2)

# Ordinal Encoding 
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/18d9f738-4373-4195-86c6-1867c0537e42)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/3c39cd75-cbc4-4be0-b6e1-77885f2157aa)

# Label Encoder
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/d6b15fec-dc0f-4335-aa5e-b7d9c8456359)

# OneHot Encoder
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
```
```
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/68e576e2-502c-4219-9c2f-82c0cc4be899)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/f725312f-3f3f-4005-8cc9-64422fd3e140)
# Binary Encoder
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/850cfb3a-1268-4876-a209-53f393e11e07)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/48f270d4-9112-4ed1-a819-966fbc9e253c)
# Target Encoder
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/4f88289a-59ac-41b4-a069-362e4ef02f9c)
# Data Transformation
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv('/content/Data_to_Transform.csv')
df
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/e534fb21-83c5-4e3d-b308-0e35d7b97b5a)
```
df.skew()
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/6fa9e262-5939-4ed8-9431-2f53e5df0249)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/ebdff44e-bf3c-4d23-8370-47de6dafc4e1)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/e1545b48-ab16-47da-84bc-f6121b46f43e)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/1d427996-1049-4286-8703-4352034c1abf)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/00c18009-d540-4d86-b440-39377bde361c)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/f3ef2b2b-9eeb-4062-888b-d118f395436b)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/ea63ad5a-cb65-4c02-8393-25bfe84571d8)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/a704ece5-fa6b-4ee8-8c2f-292747d6bcd8)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/a33675eb-6c8f-41fa-b235-c28ee2f6648b)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/977f0b7b-add6-4cff-a014-d46936459ff9)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/381b2529-69d1-449f-9268-acd090addfc1)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/d06eb8f6-3e0e-46e7-b946-262bb767b38d)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/a9f1e12e-82e6-4009-ac21-ecce9a0376cc)
```
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/fb14ea2a-4cdd-4cc3-bbe4-7c91931be99d)
```
dt=pd.read_csv("/content/titanic_dataset.csv")
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/f62a8f38-65c6-468f-820f-9a2d94bc9bec)
```
sm.qqplot(dt['Age_1'],line='45')
plt.show()
```
![image](https://github.com/23008859/EXNO-3-DS/assets/139117979/671fb915-43b8-4a79-9817-23a1cf973983)


# RESULT:
Thus perform Feature Encoding and Transformation process is executed successfully.

       
