from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout
import sys
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
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from joblib import dump, load


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.load_data = False
        self.setWindowTitle("TechBross Cars")
        self.fig = Figure()
        self.figure = self.fig.add_subplot()
        self.figure.set_title("HorsePower VS Price")
        self.figure.set_xlabel("HorsePower")
        self.figure.set_ylabel("Price (x10000 $)")

        self.fig2 = Figure()
        self.figure2 = self.fig2.add_subplot()
        self.figure2.set_title("Torque VS Price")
        self.figure2.set_xlabel("Torque")
        self.figure2.set_ylabel("Price (x10000 $)")

        self.create_widgets()

    def create_widgets(self):

        self.submit = QPushButton("Submit")
        self.submit.clicked.connect(self.click_submit)

        self.load = QPushButton("Load Data")
        self.load.clicked.connect(self.click_load)

        self.intro = QLabel("PLease Enter the Car Details")

        self.company = QLabel("Company", self)

        self.year = QLabel("Year", self)

        self.engine = QLabel("Engine Size (L) ", self)

        self.horsepower = QLabel("HorsePower", self)

        self.torque = QLabel("Torque (lb-ft)", self)

        self.time = QLabel("0-60 MPH Time (seconds)", self)

        self.companyline = QLineEdit()

        self.yearline = QLineEdit()

        self.engineline = QLineEdit()

        self.horsepowerline = QLineEdit()

        self.torqueline = QLineEdit()

        self.timeline = QLineEdit()

        self.ans = QLabel("")
        self.info = QLabel(
            "The graphs are empty because the data is not loaded.\n Load the data to see the graph")

        labels = QVBoxLayout()
        labels.addWidget(self.company)
        labels.addWidget(self.year)
        labels.addWidget(self.engine)
        labels.addWidget(self.horsepower)
        labels.addWidget(self.torque)
        labels.addWidget(self.time)

        entry = QVBoxLayout()
        entry.addWidget(self.companyline)
        entry.addWidget(self.yearline)
        entry.addWidget(self.engineline)
        entry.addWidget(self.horsepowerline)
        entry.addWidget(self.torqueline)
        entry.addWidget(self.timeline)

        main = QHBoxLayout()
        main.addLayout(labels)
        main.addLayout(entry)

        self.main_window = QVBoxLayout()

        self.main_window.addWidget(self.load)
        self.main_window.addWidget(self.intro)
        self.main_window.addLayout(main)
        self.main_window.addWidget(self.submit)
        self.main_window.addWidget(self.ans)
        self.main_window.addWidget(self.info)

        self.canvas1 = FigureCanvasQTAgg(self.fig)
        self.canvas2 = FigureCanvasQTAgg(self.fig2)
        toolbar1 = NavigationToolbar2QT(self.canvas1, self)
        toolbar2 = NavigationToolbar2QT(self.canvas2, self)

        graph = QVBoxLayout()

        graph.addWidget(toolbar1)
        graph.addWidget(self.canvas1)
        graph.addWidget(self.canvas2)
        graph.addWidget(toolbar2)

        final = QHBoxLayout()
        final.addLayout(self.main_window)
        final.addLayout(graph)

        self.setLayout(final)

    def click_submit(self):

        if self.load_data:

            if self.yearline.text() and self.engineline.text() and self.horsepowerline.text() and self.torqueline. text() and self.timeline.text():

                self.predict()
                self.update_plot()
                self.ans.setText(
                    f"Price for the car with the given attributes is : {int(self.main_ans)} $  ")
            else:
                self.ans.setText(
                    f"Please fill all the paramters correctly")

        else:
            self.ans.setText(f"First load the data")
        self.companyline.clear()
        self.yearline.clear()
        self.horsepowerline.clear()
        self.engineline.clear()
        self.torqueline.clear()
        self.timeline.clear()

    def click_load(self):
        self.load_data = True

        cars = pd.read_csv("data.csv")

        cars = cars.drop("company", axis=1)
        cars = cars.drop("model", axis=1)

        corr_matrix = cars.corr()
        print("The effect other parameters to the price is: ")
        correlation= corr_matrix['price'].sort_values(ascending=False)
        print(correlation)
        self.info.setText(f"The correlation of price with all the other parameters is : \n{correlation}\n")
        print(cars.info())
        print(cars.describe())

        data_split = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=713)
        for train_index, test_index in data_split.split(cars, cars['0to60']):
            self.train_set = cars.loc[train_index]
            test_set = cars.loc[test_index]

        train_price = self.train_set['price']
        self.train_set = self.train_set.drop("price", axis=1)
        test_price = test_set['price']
        test_set = test_set.drop("price", axis=1)

        self.my_pipeline = Pipeline(
            [
                ('imputer', SimpleImputer(strategy="median")),
                ('std_scaler', StandardScaler())
            ]
        )

        cars_train = self.my_pipeline.fit_transform(self.train_set)

        self.model = DecisionTreeRegressor()
        self.model.fit(cars_train, train_price)

        self.train_predict = self.model.predict(cars_train)

        mse = mean_squared_error(train_price, self.train_predict)
        rmse = np.sqrt(mse)
        print(f"The mean squared error of the trained data set is : {rmse}")

        x_test = self.my_pipeline.transform(test_set)
        prediction = self.model.predict(x_test)

        final_mse = mean_squared_error(test_price, prediction)
        final_rmse = np.sqrt(final_mse)
        print(f"The mean squared error of the test data set is : {final_rmse}")
        self.graph_load()

    def graph_load(self):
        self.figure.scatter(self.train_set["horsepower"], self.train_predict)

        self.figure2.bar(self.train_set["torque"],
                         self.train_predict, width=70)

        self.canvas1.draw()
        self.canvas2.draw()

    def predict(self):
        self.test = pd.DataFrame({

            'year': [float(self.yearline.text())],
            'engine_size': [float(self.engineline.text())],
            'horsepower': [float(self.horsepowerline.text())],
            'torque': [float(self.torqueline.text())],
            '0to60': [float(self.timeline.text())]
        })

        print(f"The data provided by the user is : \n{self.test}")
        self.test_ans = self.my_pipeline.transform(self.test)
        self.prediction_test = (float(self.model.predict(self.test_ans)))
        self.main_ans = self.prediction_test*10000

    def update_plot(self):

        self.figure.scatter(
            float(self.horsepowerline.text()), self.prediction_test)
        self.figure2.bar(float(self.torqueline.text()),
                         self.prediction_test, width=70)
        self.canvas1.draw()
        self.canvas2.draw()


app = QApplication([])
window = Window()
window.show()
sys.exit(app.exec())
