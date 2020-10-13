import pandas as pd
import numpy as np
import datetime
import random

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler


def load_data():
    df = pd.read_csv("diamonds.csv", index_col=0)
    return df


def data_preprocess(data):
    # Split the data into train and test
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=309)

    # Pre-process data (both train and test)
    train_data_full = train_data.copy()
    train_data = train_data.drop(["price"], axis=1)
    train_labels = train_data_full["price"]

    test_data_full = test_data.copy()
    test_data = test_data.drop(["price"], axis=1)
    test_labels = test_data_full["price"]

    cutOE = OrdinalEncoder(categories=[['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']])
    colOE = OrdinalEncoder(categories=[['J', 'I', 'H', 'G', 'F', 'E', 'D']])
    claOE = OrdinalEncoder(categories=[['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']])

    train_data['cut'] = cutOE.fit_transform(train_data_full[['cut']])
    train_data['color'] = colOE.fit_transform(train_data_full[['color']])
    train_data['clarity'] = claOE.fit_transform(train_data_full[['clarity']])
    test_data['cut'] = cutOE.transform(test_data_full[['cut']])
    test_data['color'] = colOE.transform(test_data_full[['color']])
    test_data['clarity'] = claOE.transform(test_data_full[['clarity']])

    stand = StandardScaler()
    train_data = stand.fit_transform(train_data)
    test_data = stand.transform(test_data)

    return train_data, test_data, train_labels, test_labels


def evaluate(train, train_labels, test, test_labels, model):
    start_time = datetime.datetime.now()
    model.fit(train, train_labels)
    learn_time = datetime.datetime.now() - start_time

    start_time = datetime.datetime.now()
    test_pred = model.predict(test)
    pred_time = datetime.datetime.now() - start_time

    mse = round(mean_squared_error(test_labels, test_pred),2)
    rmse = round(np.sqrt(mse),2)
    r2 = round(r2_score(test_labels, test_pred),2)
    mae = round(mean_absolute_error(test_labels, test_pred),2)
    return mse, rmse, r2, mae, round(learn_time.total_seconds() * 1000.0,2), round(pred_time.total_seconds() * 1000.0,2)


if __name__ == '__main__':
    data = load_data()
    train, test, train_labels, test_labels = data_preprocess(data)
    print(evaluate(train, train_labels, test, test_labels, LinearRegression()))
    print(evaluate(train, train_labels, test, test_labels, KNeighborsRegressor()))
    print(evaluate(train, train_labels, test, test_labels, Ridge()))
    print(evaluate(train, train_labels, test, test_labels, DecisionTreeRegressor()))
    print(evaluate(train, train_labels, test, test_labels, RandomForestRegressor()))
    print(evaluate(train, train_labels, test, test_labels, GradientBoostingRegressor()))
    print(evaluate(train, train_labels, test, test_labels, SGDRegressor()))
    print(evaluate(train, train_labels, test, test_labels, SVR(kernel='sigmoid')))
    print(evaluate(train, train_labels, test, test_labels, LinearSVR(loss='squared_epsilon_insensitive')))
    print(evaluate(train, train_labels, test, test_labels, MLPRegressor(hidden_layer_sizes=(50,100,50))))