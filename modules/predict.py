import dill
import os
import json
import pandas as pd
from datetime import datetime

path = os.environ.get('PROJECT_PATH', os.path.expanduser('~/airflow_hw'))


def predict():
    model_name = os.path.expanduser(f'{path}/data/models/cars_pipe_{datetime.now().strftime("%Y%m%d%H%M")}.pkl')
    #model_name = f'{path}/data/models/cars_pipe_202211012126.pkl'
    with open(model_name, 'rb') as file:
        model = dill.load(file)

    file_list = os.listdir(f'{path}/data/test')

    prediction_data = []
    for test_file in file_list:
        with open(os.path.join(f'{path}/data/test', test_file), 'r') as json_file:
            test_dict = json.load(json_file)
        df = pd.DataFrame([test_dict])
        prediction_data.append({'ID': test_file.replace('.json',''), 'Prediction': model.predict(df)[0]})
    prediction_df = pd.DataFrame(prediction_data)
    prediction_df.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
    predict()

