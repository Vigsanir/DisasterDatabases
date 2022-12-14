import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv(r'C:\Users\vigsa\Desktop\diplwmatikh arxeia transform\bash 8\Global_Landslide_Catalog_Export (1).csv', header=0)
data = data[[ 'landslide_size', 'fatality_count', 'injury_count']]
data.rename(columns={'landslide_size': 'disaster_Size', 'injury_count': 'injuredPeopleNumber',  'fatality_count': 'casualties'}, inplace=True)
data.casualties.replace('5,000', 5000, inplace=True)
data.casualties.replace('1,765', 1765, inplace=True)
data.casualties.replace('2,100', 2100, inplace=True)
data['casualties'] = data['casualties'].replace(['nan'], 0 )
data['injuredPeopleNumber'] = data['injuredPeopleNumber'].replace(np.nan, 0 )
data.rename(columns={'landslide_size': 'disaster_Size', 'injury_count': 'injuredPeopleNumber',  'fatality_count': 'casualties'}, inplace=True)
data.casualties.fillna(0, inplace=True)
data.casualties = data.casualties.astype('int')
le_disaster_Size = LabelEncoder()
data['disaster_Size'] = data['disaster_Size'].fillna('small' )
data['disaster_Size'].unique()
data['disaster_Size']=np.where(data['disaster_Size'] =='unknown', 'small', data['disaster_Size'])
data['disaster_Size']=np.where(data['disaster_Size'] =='nan', 'small', data['disaster_Size'])
data['disaster_Size']=np.where(data['disaster_Size'] =='very_large', 'large', data['disaster_Size'])
data['disaster_Size']=np.where(data['disaster_Size'] =='catastrophic', 'large', data['disaster_Size'])
le_disaster_Size = LabelEncoder()
data ['disaster_Size_n'] = le_disaster_Size.fit_transform(data['disaster_Size'])
X = data.drop(['disaster_Size', 'disaster_Size_n'],axis='columns')
y = data['disaster_Size_n']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)


svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)




print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

