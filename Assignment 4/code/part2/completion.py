import pandas as pd
import numpy as np
import datetime
import random

from itertools import chain

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler


def load_data():
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
     'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class']
    df = pd.read_csv("adult.data", names=names)
    df2 = pd.read_csv("adult.test", names=names, skiprows=1)
    df2['class'] = df2['class'].transform(lambda x: x[:-1])
    return df, df2


def data_preprocess(data1, data2):
    # Split the data into train and test
    train_data = data1
    test_data = data2
    train_data_full = train_data.copy()
    test_data_full = test_data.copy()

    classOE = OrdinalEncoder(categories=[[' <=50K', ' >50K']])

    train_data_full['class'] = classOE.fit_transform(train_data_full[['class']])
    test_data_full['class'] = classOE.transform(test_data_full[['class']])

    # Pre-process data (both train and test)
    train_data = train_data.drop(["class"], axis=1)
    train_labels = train_data_full["class"]

    test_data = test_data.drop(["class"], axis=1)
    test_labels = test_data_full["class"]

    eduOE = OrdinalEncoder(categories=[[' Preschool', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' 10th',
                                     ' 11th', ' 12th', ' HS-grad', ' Some-college', ' Assoc-voc', ' Assoc-acdm',
                                     ' Bachelors', ' Masters', ' Prof-school', ' Doctorate']])

    train_data['education'] = eduOE.fit_transform(train_data_full[['education']])
    test_data['education'] = eduOE.transform(test_data_full[['education']])

    enc = OneHotEncoder()
    X = train_data[['workclass', 'marital-status', 'relationship', 'race', 'sex', 'native-country', 'occupation']]
    Y = test_data[['workclass', 'marital-status', 'relationship', 'race', 'sex', 'native-country', 'occupation']]
    tr = enc.fit_transform(X)
    te = enc.transform(Y)
    cats = list(chain.from_iterable(enc.categories_))
    train_data = train_data.join(pd.DataFrame(tr.toarray(), columns=cats))
    train_data = train_data.drop(['workclass', 'marital-status', 'relationship', 'race', 'sex', 'native-country', 'occupation'], axis=1)
    test_data = test_data.join(pd.DataFrame(te.toarray(), columns=cats))
    test_data = test_data.drop(['workclass', 'marital-status', 'relationship', 'race', 'sex', 'native-country', 'occupation'], axis=1)

    stand = StandardScaler()
    train_data = stand.fit_transform(train_data)
    test_data = stand.transform(test_data)

    return train_data, test_data, train_labels, test_labels


def evaluate(train, train_labels, test, test_labels, model):
    model.fit(train, train_labels)
    test_pred = model.predict(test)

    accuracy = accuracy_score(test_labels, test_pred)
    precision = precision_score(test_labels, test_pred)
    recall = recall_score(test_labels, test_pred)
    f1 = f1_score(test_labels, test_pred)
    auc = roc_auc_score(test_labels, test_pred)
    return round(accuracy,2), round(precision,2), round(recall,2), round(f1,2), round(auc,2)


if __name__ == '__main__':
    data1, data2 = load_data()
    train, test, train_labels, test_labels = data_preprocess(data1, data2)
    print(evaluate(train, train_labels, test, test_labels, KNeighborsClassifier()))
    print(evaluate(train, train_labels, test, test_labels, GaussianNB()))
    print(evaluate(train, train_labels, test, test_labels, LinearSVC()))
    print(evaluate(train, train_labels, test, test_labels, DecisionTreeClassifier(criterion='entropy')))
    print(evaluate(train, train_labels, test, test_labels, RandomForestClassifier(n_estimators=100)))
    print(evaluate(train, train_labels, test, test_labels, AdaBoostClassifier()))
    print(evaluate(train, train_labels, test, test_labels, GradientBoostingClassifier(n_estimators=500)))
    print(evaluate(train, train_labels, test, test_labels, LinearDiscriminantAnalysis()))
    print(evaluate(train, train_labels, test, test_labels, MLPClassifier(activation='logistic')))
    print(evaluate(train, train_labels, test, test_labels, LogisticRegression()))