### Singh, Bhanu Pratap  22201990

### Galani, Rohit   22209032


# TechBros Car Price Predictor

[Link to MyGit repository](https://mygit.th-deg.de/techbros/techbroscars.git)


# Project Description
The "TechBros Car Price Predictor" is a PyQt6-based desktop application designed to help users predict the price of a car based on various input parameters and visualize the relationships between different car attributes and their impact on prices. The tool utilizes machine learning algorithms, specifically a Decision Tree Regressor, to predict car prices and displays the results through interactive graphs


# Prerequisites
- Python [3.11.2]
- PyQt6 [6.6.1]
- Pandas [2.2.0]
- Numpy [1.24.3]
- Scikit-learn [1.4.0]
- Matplotlib [3.8.2]
- Joblib [1.3.2]



# Installation
To run this chatbot properly, follow the given steps : 
- Download or clone this repository:
```
git clone https://mygit.th-deg.de/techbros/techbroscars.git
```
- Now, you must open this porject in a virtual python environment using venv module 
```
cd rasa-chatbot-main 
pip install virtualenv
```
- Create a new virtual environment (Python 3.11.2) in the working directory and activate it
```
python -m venv env

.\env\Scripts\activate.bat (on windows)
```
- Then, you have to run the following commands to install all the required modules :

```
pip install numpy, pandas,PyQt6, scikit-learn, matplotlib, joblib
```
# Basic Usage

- To use this app, simply run the mainProject.py 
```
cd techbroscars-main
python mainProject.py
```
After opening the app, 
- Firstly, user must load the data by clicking on the button "Load Data"
- User can simply enter parameters like Company, Year of manufacture, engine size, horsepower, torque, 0 to 60 mph time.
- The predicted price will be shown once the entered data is submitted
Example : 
          
          Company - Porsche

          Year - 2022

          Engine Size(L) - 3

          horsepower - 379 

          torque(lb-ft) - 331

          0to60 mph time (seconds) - 4

    **Predicted Price** - 146570 $
# Implementation of the Requests

Overview of the implementation of the methods used in this GUI window class : 

1. **\_\_init\_\_ method (Constructor):**
   - Initializes the main window of the application.
   - Sets up Matplotlib figures and subplots for data visualization.
   - Creates various widgets, labels, and layout structures for the user interface.

2. **create_widgets method:**
   - Defines and organizes the layout of input widgets, labels, buttons, and information labels within the main window.
   - Utilizes QVBoxLayout and QHBoxLayout to arrange the UI elements.

3. **click_submit method:**
   - Handles the logic when the "Submit" button is clicked.
   - Checks if the data is loaded and all necessary parameters are provided.
   - Calls the predict method to make a prediction based on user input.
   - Updates the data visualization plots using update_plot method.
   - Displays the predicted price as a message.

4. **click_load method:**
   - Loads the dataset from the "data.csv" file using Pandas read_csv.
   - Performs data preprocessing by dropping unnecessary columns and splitting the data into training and testing sets.
   - Sets up a machine learning pipeline (imputation and scaling) and trains a Decision Tree Regressor model.
   - Calculates and prints mean squared error for both the training and test datasets.
   - Calls graph_load to initialize and update the initial data visualization.

5. **graph_load method:**
   - Populates the Matplotlib figures with initial data for horsepower vs. price and torque vs. price.
   - Draws the initial scatter and bar plots for visualization.
   - Called during data loading to provide an initial representation of the dataset.

6. **predict method:**
   - Creates a DataFrame with user-provided input for making a prediction.
   - Transforms the user input using the pre-defined pipeline.
   - Utilizes the trained Decision Tree Regressor to predict the car price.
   - Stores the predicted result for later display.

7. **update_plot method:**
   - Updates the data visualization plots based on user input and predictions.
   - Modifies the scatter and bar plots with the new data.
   - Draws the updated plots to reflect the changes interactively.

These methods collectively implement the functionality required for data loading, preprocessing, machine learning model training, user input handling, and real-time data visualization in the PyQt6 application.

# Work done

## Bhanu Pratap Singh : (22201990)

1) Graphical User Interface
2) Pandas with Numpy
3) General Python Programming

## Rohit Galani : (22209032)

1) Visualization
2) Scikit-Learn
3) General Python Programming

