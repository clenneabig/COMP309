from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor
import pandas
import numpy
import matplotlib.pyplot
import seaborn



def filtermethod():
    df = pandas.read_csv("electricity_data_labelled.csv")
    """df["Price"] = x.target
    X = df.drop("Price", 1)   #Feature Matrix
    y = df["Price"]          #Target Variable
    df.head()"""
    # Using Pearson Correlation
    # plt.figure(figsize=(50,48))
    cor = df.corr()
    # sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    # plt.show()
    cor_target = abs(cor["Price"])
    # Selecting highly correlated features
    relevant_features = cor_target[cor_target > 0.5]
    print(relevant_features)
    """print(df[["arapuni","atiamuri"]].corr())
    print(df[["arapuni","huntly_1_4"]].corr())
    print(df[["arapuni","maraetai"]].corr())
    print(df[["arapuni","ohakuri"]].corr())
    print(df[["arapuni","waipapa"]].corr())
    print(df[["arapuni","waipori"]].corr())
    print(df[["arapuni","whakamaru"]].corr())
    print(df[["atiamuri","huntly_1_4"]].corr())
    print(df[["atiamuri","maraetai"]].corr())
    print(df[["atiamuri","ohakuri"]].corr())
    print(df[["atiamuri","waipapa"]].corr())
    print(df[["atiamuri","waipori"]].corr())
    print(df[["atiamuri","whakamaru"]].corr())
    print(df[["hun","huntly_1_4"]].corr())"""
    matplotlib.pyplot.figure(figsize=(12, 10))
    seaborn.heatmap(df[["huntly_1_4", "ohakuri", "waipori"]].corr(), annot=True, cmap=matplotlib.pyplot.cm.Reds)
    matplotlib.pyplot.show()


training = pandas.read_csv("electricity_data_labelled.csv")
test = pandas.read_csv("electricity_data_unlabelled.csv")

training.fillna(0, inplace=True)
test.fillna(0, inplace=True)
beforelen = len(training)

iso = IsolationForest(n_estimators=10)
outliers_train = iso.fit_predict(training)
training = training[numpy.where(outliers_train == 1, True, False)]

afterlen = len(training)
print(beforelen-afterlen)

training.pop("Id")
price = training.pop("Price")
ids = test.pop("Id")

imp = SimpleImputer()
rf = RandomForestRegressor()

pipe = Pipeline(steps=[("imp", imp), ('rf', rf)])

pipe.fit(training, price)
predictions = pipe.predict(test)

pred_table = pandas.DataFrame(predictions, columns=['Price'])
output = pandas.concat([ids, pred_table], axis=1, sort=False)
output.to_csv('predictions.csv', index=False)


