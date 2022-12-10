'''

4개의 Classifier를 pipeline으로 연결해서 성능 확인

'''

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pickle
import pandas as pd

df = pd.read_csv('coords.csv')
X = df.iloc[:,1:]
y = df.iloc[:,0]

X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=3213
)

pipelines = {
    'lr' : make_pipeline(StandardScaler(), LogisticRegression()),
    'rc' : make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf' : make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb' : make_pipeline(StandardScaler(), GradientBoostingClassifier())
}

fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model

for algo, model in fit_models.items():
    result = model.predict(X_test)
    print(algo, accuracy_score(y_test, result))

# 모델 저장
with open('face_rf.pkl','wb') as f:
    pickle.dump(fit_models['rf'], f)