import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

data = pd.read_csv(r'C:\Users\vigsa\Desktop\diplwmatikh arxeia transform\bash 8\Global_Landslide_Catalog_Export (1).csv', header=0)
data = data[[ 'landslide_size', 'fatality_count', 'injury_count']]
data.rename(columns={'landslide_size': 'disaster_Size', 'injury_count': 'injuredPeopleNumber',  'fatality_count': 'casualties'}, inplace=True)
data.casualties.replace('5,000', 5000, inplace=True)
data.casualties.replace('1,765', 1765, inplace=True)
data.casualties.replace('2,100', 2100, inplace=True)
data['casualties'] = data['casualties'].replace(['nan'], 0 )
data['injuredPeopleNumber'] = data['injuredPeopleNumber'].replace(np.nan, 0 )
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
X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=7) 
knn.fit(X_train, y_train)

print(knn.predict(X_test))

neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
  
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
  
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm,annot=True)

plt.savefig('confusion_Matrix.png')
print(cm)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

