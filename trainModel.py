import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from joblib import dump

cars = pd.read_csv("data.csv")

cars = cars.drop("company", axis=1)
cars = cars.drop("model", axis=1)

corr_matrix = cars.corr()
print("The effect other parameters to the price is: ")
print(corr_matrix['price'].sort_values(ascending=False))
print(cars.info())
print(cars.describe())


data_split = StratifiedShuffleSplit(
    n_splits=1, test_size=0.2, random_state=713)
for train_index, test_index in data_split.split(cars, cars['0to60']):
    train_set = cars.loc[train_index]
    test_set = cars.loc[test_index]

train_price = train_set['price']
train_set = train_set.drop("price", axis=1)
test_price = test_set['price']
test_set = test_set.drop("price", axis=1)



my_pipeline = Pipeline(
    [
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler())
    ]
)

cars_train = my_pipeline.fit_transform(train_set)

model = DecisionTreeRegressor()
model.fit(cars_train, train_price)

train_predict = model.predict(cars_train)

mse = mean_squared_error(train_price, train_predict)
rmse = np.sqrt(mse)
print(f"\nThe mean squared error of the trained data set is : {rmse}\n")


x_test = my_pipeline.transform(test_set)
prediction = model.predict(x_test)

final_mse = mean_squared_error(test_price, prediction)
final_rmse = np.sqrt(final_mse)
print(f"The mean squared error of the test data set is : {final_rmse}\n")


print("Five sample test data correct price vs the predicted price (x10000 $)")
for i in range(5):
    print(str(test_price.iloc[i])+' --- '+str(prediction[i]))


dump(model, "TechBross Cars.joblib")
