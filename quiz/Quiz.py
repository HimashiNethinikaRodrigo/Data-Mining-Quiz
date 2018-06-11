import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as final_score
from sympy import preorder_traversal

df = pd.read_csv('./TrainingData.csv', infer_datetime_format=True)
df_test= pd.read_csv('./TestDataWithHeaders.csv')
np.sum(df.isnull(), axis=0)
df = df.dropna(axis=0, thresh=20)
np.sum(df.isnull(), axis=0)
print(df.Class.value_counts())
x = df.drop(["Class"], axis=1)
y= df["Class"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=4)
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(x_train, y_train)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                           oob_score=False, random_state=None, verbose=0,
                           warm_start=False)


score = rf.score(x_test, y_test)
print("Random Forest: ", score)
y_score = rf.predict(x_test)
test_precision, recall, fscore, support = final_score(y_test, y_score)
print("test precision: ",test_precision )
print("test recall: ", recall)
print("test support: ", support)
print("test score: ", fscore)
value= rf.predict(df_test)
for i in value:
    print(i)