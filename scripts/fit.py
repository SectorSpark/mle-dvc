import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import yaml
import os
import joblib

# обучение модели
def fit_model():
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)
        
    # загрузите результат предыдущего шага: inital_data.csv
    data = pd.read_csv('data/initial_data.csv')
    
    # реализуйте основную логику шага с использованием гиперпараметров
    cat_features = data.select_dtypes(include='object')
    num_features = data.select_dtypes(['float64', 'int64'])
    
    preprocessor = ColumnTransformer(
        [
            ('cat', OneHotEncoder(drop='if_binary'), cat_features.columns.tolist()),
            ('num', StandardScaler(), num_features.columns.tolist())
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    
    model = LogisticRegression(C=params['C'], penalty=params['penalty'], max_iter=1000)
    
    pipeline = Pipeline(
        [
            ('preprocessor', preprocessor),
            ('model', model)
        ]
    )
    pipeline.fit(data, data['target'])
    
    # сохраните обученную модель в models/fitted_model.pkl
    os.makedirs('models', exist_ok=True)
    with open('models/fitted_model.pkl', 'wb') as fd:
        joblib.dump(pipeline, fd)

if __name__ == '__main__':
    fit_model()
