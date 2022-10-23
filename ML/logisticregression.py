import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
data = pd.read_csv(r'C:\Users\vigsa\Desktop\diplwmatikh arxeia transform\bash 8\Global_Landslide_Catalog_Export (1).csv', header=0)
data = data[[ 'landslide_size', 'fatality_count', 'injury_count']]
data.rename(columns={'landslide_size': 'disaster_Size', 'injury_count': 'injuredPeopleNumber',  'fatality_count': 'casualties'}, inplace=True)
data['disaster_Size'].unique
data['disaster_Size']=np.where(data['disaster_Size'] =='unknown', 'small', data['disaster_Size'])
data['disaster_Size']=np.where(data['disaster_Size'] =='nan', 'small', data['disaster_Size'])
data['disaster_Size']=np.where(data['disaster_Size'] =='very_large', 'large', data['disaster_Size'])
data['disaster_Size']=np.where(data['disaster_Size'] =='catastrophic', 'large', data['disaster_Size'])
data['disaster_Size'].unique
data['casualties'] = data['casualties'].replace(['nan'], 0 )
data['injuredPeopleNumber'] = data['injuredPeopleNumber'].replace(np.nan, 0 )
data.casualties.replace('5,000', 5000, inplace=True)
data.casualties.replace('1,765', 1765, inplace=True)
data.casualties.replace('2,100', 2100, inplace=True)
data.casualties.fillna(0, inplace=True)
data.casualties = data.casualties.astype('int')
le_disaster_Size = LabelEncoder()
data['disaster_Size'] = data['disaster_Size'].fillna('small' )
data ['disaster_Size_n'] = le_disaster_Size.fit_transform(data['disaster_Size'])
data_n = data.drop(['disaster_Size'],axis='columns')
target = data['disaster_Size_n']
data_n = data.drop(['disaster_Size_n', 'disaster_Size'],axis='columns')
data_n_train, data_n_test, target_train, target_test = train_test_split(data_n, target, test_size=0.25, random_state=1)
log_reg = LogisticRegression()
log_reg.fit(data_n_train, target_train)
print(log_reg.coef_)
print(log_reg.intercept_)
target_pred = log_reg.predict(data_n_test)
confusion_matrix(target_test, target_pred)
logreg = LogisticRegression(random_state=16)
logreg.fit(data_n_train, target_train)
target_pred = logreg.predict(data_n_test)
plt.scatter(data_n, target, c=target, cmap='rainbow')
plt.title('Scatter Plot of Logistic Regression')
plt.show()
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(target_test, target_pred)
cnf_matrix
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

