# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:47:03 2020

@author: shadi
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

st.title('Titanic Survivors')

st.subheader('Would you survive the wreckage of the Titanic?')

DATE_COLUMN = 'date/time'
DATA_TRAIN = 'train.csv'

PLOT_TYPE = {
    'Pie chart': 'pie',
    'Bar chart': 'bar',
    'Scatter plot': 'scatter',
    'Line plot': 'line',
    'Histogramm': 'hist',
    'Boxplot': 'box',
    'KDE plot': 'kde',
    'Area plot': 'area',
    'Hexbin plot': 'hexbin'
    }

@st.cache
def load_data():
    data = pd.read_csv(DATA_TRAIN)
    data_p = data.copy()
    data_p =data_p.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
    values = {'Age': data_p['Age'].mean(), 'Cabin': 'NO CABIN', 'Embarked': data_p['Embarked'].mode()[0]}
    data_p = data_p.fillna(value=values)
    data_e = data_p.copy()
    data_e = pd.get_dummies(data_e, columns=['Sex'], prefix = ['Sex'], drop_first=True)
    data_e = pd.get_dummies(data_e, columns=['Embarked'], prefix = ['Embarked'], drop_first=True)
    data_e['Cabin Owner'] = np.where(data_e['Cabin'].str.contains('NO CABIN'), 0, 1)
    data_e = data_e.drop(['Cabin'], axis=1)
    return (data, data_p, data_e)

data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text("Done! (using st.cache)")

DATA_STATE = {
    'Raw data': 0,
    'Preprocessed data': 1,
    'Encoded data': 2
    }

state_data = st.selectbox("Choose which data to display: ",list(DATA_STATE.keys()))
if st.checkbox('Show data'):
    st.subheader(f'{state_data}:')
    st.write(data[DATA_STATE[state_data]])  

current_data = data[DATA_STATE[state_data]]

#Correlation plot
st.subheader('Correlation plot')
if st.button("Show correlation plot?"):
    corrplot= sns.heatmap(current_data.corr(), annot=True)
    st.write(corrplot)
    st.pyplot()

st.subheader('Plotting data')
type_plot = st.selectbox("How do you want to plot the data", list(PLOT_TYPE.keys()))
selected_cols = st.multiselect("Select columns to plot", current_data.columns.values)

if st.button("Generate plot"):
    st.success(f"Generating plot of {type_plot} for {selected_cols}")
    req_data = current_data[selected_cols].plot(kind=PLOT_TYPE[type_plot])
    st.write(req_data)
    st.pyplot()
    
## MACHINE LEARNING    
st.subheader('Selecting a model')

MODEL_LIST = {
        'KNN': neighbors.KNeighborsClassifier(n_neighbors = 5,p=1,weights="distance"),
        'SVC': SVC(gamma='auto'),
        'Random forest': RandomForestClassifier(),
        'Gaussian Naive Bayes': GaussianNB(),
        'Multinomial NB': MultinomialNB(),
        'Logistic regression': LogisticRegression()
    }

study_set = data[2]
target_variable = st.selectbox("Which variable would you like to select as your target?",
                               study_set.columns.values
                               )
y = study_set[target_variable]
X = study_set.copy()
X = study_set.drop(target_variable, axis=1)

req_random_state = st.slider("Pick a random state to split the model", 0, 100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,  random_state=req_random_state)

req_model = st.selectbox("Choose a ML model", list(MODEL_LIST.keys()))
curr_model = MODEL_LIST[req_model]
scores = cross_val_score(curr_model, X, y, cv=5)

st.success(f"Scores calculated for {req_model}")
st.text("Average accuracy: {}".format(np.mean(scores)))

# Adding a fit
curr_model.fit(X_train, y_train)

# Predictions
y_pred = curr_model.predict(X_test)

if st.button("See predictions in detail?"):
    req_results = {
        'Test' : list(y_test),
        'Predicted' : list(y_pred)
        }
    st.write(pd.DataFrame(req_results))

st.subheader("How do you want to evaluate the model's performance?")

VAL_METHOD = {
    'Accuracy Score': accuracy_score(y_test, y_pred),
    'Classification report': classification_report(y_test, y_pred),
    'Confusion matrix': confusion_matrix(y_test, y_pred)
    }

req_val_method = st.radio(
    "Select a classification metric",
     tuple(VAL_METHOD.keys()))

curr_val_method = VAL_METHOD[req_val_method]

if req_val_method == 'Accuracy Score':
    st.write(f"The Accuracy Score is {curr_val_method}")
    st.write("Number of matches: ", 
             accuracy_score(y_test, y_pred, normalize=False), "over",
             y_test.size)
elif req_val_method == 'Classification report':
    # Need to transform this dictionary into a pd df object
    st.write(curr_val_method)
elif req_val_method == 'Confusion matrix':
    cm_df = pd.DataFrame(curr_val_method, 
                         index=["Predicted +", "Predicted -"],
                         columns=["Actual +", "Actual -"])
    #st.write(cm_df)
    cm_plot = sns.heatmap(cm_df, annot=True)
    st.write(cm_plot)
    st.pyplot()
    
#Insert GridSearchCV here
st.subheader("Model tuning")

#Use this object to refer the model tuning
HYPER_PARAM = {
    'SVC': {'kernel':('linear', 'rbf'), 'C':[1, 10]},
    'KNN': {'n_neighbors': list(range(4,12)), 
            'weights':('uniform', 'distance')},
    'Random forest': {'n_estimators': list(range(90,110)), 
                      'criterion': ('gini','entropy')},
    'Gaussian Naive Bayes': {},
    'Multinomial NB': {},
    'Logistic regression': {'penalty':('l1', 'l2', 'elasticnet'), 
                            'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    }

if req_model == 'Multinomial NB' or req_model == 'Gaussian Naive Bayes':
    st.text("No model available for this method.")
else:
    curr_param = HYPER_PARAM[req_model]
    gscv = GridSearchCV(curr_model, curr_param)
    gscv.fit(X_train, y_train)
    if st.button("Show my results!"):
        st.text("Best parameters")
        st.write(gscv.best_params_)
        st.text("Best estimator")
        st.write(gscv.best_estimator_)
        st.text("The results overall...")
        st.write(gscv.cv_results_)
    

## MODEL TEST WITH 1 INDIVIDUAL
st.header("Introduce yourself!")

x_individual = {}
x_individual["Pclass"] = st.slider('Are you rich?', 1, 3, 2)
x_individual["Age"] = st.slider('How old are you?', 0, 130, 25)
x_individual["SibSp"] = st.slider('Any siblings aboard?', 0, 5)
x_individual["Parch"] = st.slider('Any kids aboard or did you bring a parent?', 0, 5)
x_individual["Fare"] = st.slider('Gave a tip?', 0, 100)
x_individual["Sex_male"] = st.slider('Are you male? (1 for yes)', 0, 1)
embarkation = st.radio("Where would you embark?", ('Cherbourg', 'Queenstown', 'Southhampton'))

if embarkation == 'Queenstown':
    x_individual["Embarked_Q"] = 1
    x_individual["Embarked_S"] = 0
elif embarkation == 'Southhampton':
    x_individual["Embarked_Q"] = 0
    x_individual["Embarked_S"] = 1
else:
    x_individual["Embarked_Q"] = 0
    x_individual["Embarked_S"] = 0
    
x_individual["Cabin Owner"] = st.slider('Did you rent a cabin? (0 for no)', 0, 1)

x_individual = pd.DataFrame(x_individual, index=[0])

y_survived = curr_model.predict(x_individual)
ind_proba = curr_model.predict_proba(x_individual)

x_name = st.text_input("Finally what's your full name?", 
                                     'John Doe')

if st.button("Did you make it?"):
    honorific = "Mr" if x_individual['Sex_male'][0]==1 else "Mrs"
    st.text(f"Dear {honorific}. {x_name}...")
    st.text(f"With {ind_proba} of chance...")
    if y_survived == 0:
        st.text("You died in the shipwreck.\n May you rest in peace")
    else:
        st.text("You survived! Congratulations!")
    