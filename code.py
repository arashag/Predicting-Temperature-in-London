# Run this cell to import the modules you require
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

# Read in the data
weather = pd.read_csv("london_weather.csv",parse_dates=['date'])

# Extracting features from date field
weather['year']=weather['date'].dt.year
weather['month']=weather['date'].dt.month

#Dropping null target field records
weather=weather.dropna(subset=['mean_temp'])

# Splitting to train and test sets
X=weather.drop(columns=['mean_temp'])
y=weather['mean_temp']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=42)
print((X_train.shape,y_train.shape),(X_test.shape,y_test.shape))

#Building training pipeline
mlflow.set_experiment('london_mean_temp')
mlflow.autolog()


model_dict={'lin_reg':LinearRegression(),
            'dt_tree':DecisionTreeRegressor(),
            'rn_forest':RandomForestRegressor(random_state=42)}

col_trans=ColumnTransformer([('imputation',
                          SimpleImputer(strategy='median'),
                          ['cloud_cover', 'sunshine', 'global_radiation', 'precipitation', 'pressure', 'snow_depth','year', 'month'])])

for model_name,model in model_dict.items():

    with mlflow.start_run(run_name=model_name):

        mlflow.set_tag("model_type", model_name)
        train_pipeline=Pipeline([('transformer',col_trans),
                                 ('scaler',StandardScaler()),
                                 (f'{model_name}',model)])

        #train_pipeline.fit(X_train,y_train)

        param_grids = {
            "lin_reg": {},
            "dt_tree": {f"{model_name}__max_depth": range(1,11)},
            "rn_forest": {f"{model_name}__max_depth": range(1,11)},
        }   

        grid = GridSearchCV(train_pipeline, param_grids[model_name],scoring='neg_root_mean_squared_error', cv=5)

        grid.fit(X_train,y_train)
        
        mlflow.sklearn.log_model(sk_model=grid.best_estimator_,
                                 artifact_path=f"{model_name}",
                                 registered_model_name=f"best_{model_name}")

        rmse=mean_squared_error(y_test,grid.best_estimator_.predict(X_test),squared=False)

        mlflow.log_metric(f'best_rmse',rmse)

experiment_results=mlflow.search_runs()    


