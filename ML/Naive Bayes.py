import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import numpy as np

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
data ['disaster_Size_n'] = le_disaster_Size.fit_transform(data['disaster_Size'])
data_n = data.drop(['disaster_Size'],axis='columns')
target = data['disaster_Size_n']
data_n = data.drop(['disaster_Size_n', 'disaster_Size'],axis='columns')
X = data.drop(['disaster_Size', 'disaster_Size_n'],axis='columns')
y = data['disaster_Size_n']
prior = data.groupby('disaster_Size_n').size().div(len(data)) 
print(prior)
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
print(clf.predict([[-0.8, -1]]))
clf_pf = GaussianNB()
clf_pf.partial_fit(X, y, np.unique(y))
GaussianNB()
print(clf_pf.predict([[-0.8, -1]]))
fit(X, y, sample_weight=None)
data.sort_index()

Vectorizer = CountVectorizer(stop_words)
all_features = vectorizer.fir_transform(data.disaster_ize)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3,random_state=0)
clf = GaussianNB()
clf.fit(X_train, y_train)
GaussianNB()
y_pred=clf.predict(X_test)
print(classification_report(y_test,y_pred))

