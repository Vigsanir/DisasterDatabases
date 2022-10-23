import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
model = DecisionTreeClassifier()
df = pd.read_csv(r'C:\Users\vigsa\Desktop\diplwmatikh arxeia transform\bash 8\Global_Landslide_Catalog_Export (1).csv')
df = df[['event_date', 'event_time','event_title', 'event_description', 'location_description', 'landslide_size', 
        'fatality_count', 'injury_count', 'country_name', 'country_code', 'longitude', 'latitude']]
df['year'] = df.event_date.apply(lambda x: x[6: 11])
df['database'] = 'Global Landslide Catalog'
df['startYear'] = df.event_date.apply(lambda x: x[6: 10])
df['startMonth'] = df.event_date.apply(lambda x: x[3: 5])
df['startDay'] = df.event_date.apply(lambda x: x[: 2])
df.rename(columns={'event_description': 'description', 'location_description': 'location', 'fatality_count': 'casualties'
    , 'injury_count': 'injuredPeopleNumber', 'country_name': 'country'}, inplace=True)
df.rename(columns={'location_description': 'location'}, inplace=True)
df['startTime'] = df.event_date.apply(lambda x: x[11: ])
df['startDate'] = df.event_date.apply(lambda x: x[:10])
df.drop(columns=['event_date', 'event_time', 'event_title', 'country_code', ], inplace=True)
df.rename(columns={'landslide_size': 'disaster_Size', 'description': 'event_Description'}, inplace=True)
df.drop(columns=['year'], inplace=True)
df.rename(columns={'country_name': 'country'}, inplace=True)
X = df[['disaster_Size', 'casualties', 'injuredPeopleNumber']]
X.rename(index = {'unknown': 'small'})
X.rename({'unknown': 'small'})
X['disaster_Size'].unique()
X['disaster_Size'] = X['disaster_Size'].replace(['nan'], 'small' )
X['disaster_Size'] = X['disaster_Size'].replace(['unknows'], 'small' )
X['disaster_Size'] = X['disaster_Size'].replace(['very_large'], 'large' )
X['disaster_Size'] = X['disaster_Size'].replace(['catastrophic'], 'large' )
X['disaster_Size'] = X['disaster_Size'].replace(['unknown'], 'small' )
X['disaster_Size'] = X['disaster_Size'].fillna('small' )
y= X['injuredPeopleNumber']
k =X
le_disaster_Size = LabelEncoder() 
X ['disaster_Size_n'] = le_disaster_Size.fit_transform(X['disaster_Size'])
X.head()
X_n = X.drop(['disaster_Size'],axis='columns')
target = X['disaster_Size_n']
X_n = X.drop(['disaster_Size_n', 'disaster_Size'],axis='columns')

X_n['casualties'] = X_n['casualties'].replace(['nan'], 0 )
X_n['injuredPeopleNumber'] = X_n['injuredPeopleNumber'].replace(np.nan, 0 )
X_n['injuredPeopleNumber'].nunique()
X_n.casualties.fillna(0, inplace=True)
X_n.casualties.replace('5,000', 5000, inplace=True)
X_n.casualties.replace('1,765', 1765, inplace=True)
X_n.casualties.replace('2,100', 2100, inplace=True)
X_n.casualties = X_n.casualties.astype('int')
from sklearn import tree
model = tree.DecisionTreeClassifier(criterion='gini', max_depth=50)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_n, target, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, preds)
logreg = LogisticRegression(random_state=16)
logreg.fit(data_n_train, target_train)
target_pred = logreg.predict(data_n_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(X_test, X_pred)
cnf_matrix

