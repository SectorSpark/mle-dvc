# scripts/evaluate.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib
import json
import yaml
import os

# оценка качества модели
def evaluate_model():
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)
	# прочитайте файл с гиперпараметрами params.yaml
    pipeline = joblib.load('models/fitted_model.pkl') #models/fitted_model.pkl
	# загрузите результат прошлого шага: fitted_model.pkl
    # Проверка качества на кросс-валидации
    data = pd.read_csv('data/initial_data.csv')
    cv_strategy = StratifiedKFold(n_splits=5)
    cv_res = cross_validate(
        pipeline,
        data,
        data['target'],
        cv=cv_strategy,
        n_jobs=-1,
        scoring=['f1', 'roc_auc']
        )
    for key, value in cv_res.items():
        cv_res[key] = round(value.mean(), 3)
    os.makedirs('cv_results', exist_ok=True)
	# реализуйте основную логику шага с использованием прочтённых гиперпараметров
    with open('cv_results/cv_res.json', 'w') as json_file:
        json.dump(cv_res, json_file, indent=4)
	# сохраните результата кросс-валидации в cv_res.json


if __name__ == '__main__':
	evaluate_model()