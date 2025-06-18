import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

dataset = pd.read_csv('processed_data_1.csv')

x_df = dataset.iloc[:, 1:-1] 
y = dataset.iloc[:,89].values

non_num_cols = x_df.select_dtypes(exclude=['int64', 'float64']).columns
for col in non_num_cols:
    x_df[col] = pd.to_numeric(x_df[col], errors='coerce')
x_imputer = SimpleImputer(missing_values=np.nan,strategy='mean', fill_value=None)
x_imputed = imputer.fit_transform(x_df)
x = pd.DataFrame(x_imputed, columns=x_df.columns)

y = y.reshape(-1, 1)
y_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
y = y_imputer.fit_transform(y)
y = y.ravel()
y = y.astype(int) 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=300,criterion='entropy' , random_state= 0)
classifier.fit(x_train,y_train)

importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]
top_features = indices[:35]  

x_train_sel = x_train[:, top_features]
x_test_sel = x_test[:, top_features]

final_clf = RandomForestClassifier(n_estimators=300, criterion='entropy', random_state=0)
final_clf.fit(x_train_sel, y_train)


y_pred = final_clf.predict(x_test_sel)
accuracy = accuracy_score(y_test, y_pred)
print("使用前35特徵後的準確率:", accuracy)

print("\n分類報告（Classification Report）:")
print(classification_report(y_test, y_pred, digits=4))


print("混淆矩陣:")
print(confusion_matrix(y_test, y_pred))

basic_features = ['C-reactive protein (mg/dL)', 'WBC (x10^3/uL)', '心跳(H)','PT(second)', 'GOT(AST) (U/L)']
print("\n患者基本資料描述性統計:\n")
summary = dataset[basic_features + ['Y']].groupby('Y').describe().T
print(summary)

plt.figure(figsize=(8, 6))
sns.boxplot(x='Y', y='WBC\xa0(x10^3/uL)', data=dataset)
plt.title('WBC Distribution')
plt.xlabel('Result (Y)')
plt.ylabel('WBC (x10^3/uL)')
plt.grid(True)
plt.show()


#C-reactive protein (mg/dL)
dataset['C-reactive protein\xa0(mg/dL)'] = pd.to_numeric(dataset['C-reactive protein\xa0(mg/dL)'], errors='coerce')
q1 = dataset['C-reactive protein\xa0(mg/dL)'].quantile(0.25)
q3 = dataset['C-reactive protein\xa0(mg/dL)'].quantile(0.75)
iqr = q3 - q1

lower = max(0, q1 - 1.5 * iqr)
upper = q3 + 1.5 * iqr  

plt.figure(figsize=(8, 6))
sns.boxplot(x='Y', y='C-reactive protein\xa0(mg/dL)', data=dataset)
plt.title('C-reactive protein (mg/dL) Distribution') 
plt.ylabel('C-reactive protein (mg/dL)')
plt.xlabel('Result (Y)')
plt.ylim(lower, upper)
plt.grid(True)
plt.show()

#GOT(AST)
q1 = dataset['GOT(AST)\xa0(U/L)'].quantile(0.25)
q3 = dataset['GOT(AST)\xa0(U/L)'].quantile(0.75)
iqr = q3 - q1

lower = max(0, q1 - 1.5 * iqr)
upper = q3 + 1.5 * iqr 

plt.figure(figsize=(8, 6))
sns.boxplot(x='Y', y='GOT(AST)\xa0(U/L)', data=dataset)
plt.title('GOT(AST) (U/L) Distribution') 
plt.ylabel('GOT(AST) (U/L)')
plt.xlabel('Result (Y)')
plt.ylim(lower, upper)
plt.grid(True)
plt.show()









