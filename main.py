import streamlit as st
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

@st.cache_data
def get_data(filename):
    wine=pd.read_csv(filename,decimal=',',delimiter=";")
    return wine

with header:
    st.title('Welcome to my first streamlit website')
    st.text('Dans ce projet nous allons étudier la qualité du vin')

with dataset:
    st.header('wine quality dataset')
    wine=get_data('data/winequality-red.csv')
    st.write(wine.head())

    st.subheader('Fixed acidity distribution')
    fixed_acidity_dist=pd.DataFrame(wine['fixed acidity'].value_counts())
    st.bar_chart(fixed_acidity_dist)

with features:
    st.header('The features')

    st.markdown('* **first features:**')

with model_training:
    st.header('Time to train the model')
    st.text('this part is dedicated to choose hyperparametrs to train our model')
    
    sel_col, disp_col=st.columns(2)
    max_depth=sel_col.slider('what should be the max depth of the model,',min_value=3,max_value=20,step=1)
    n_estimators=sel_col.selectbox('how many trees should there be',options=[100,200,300,'No limit'])

    sel_col.text('Liste des features pouvant etre selectionnée')
    sel_col.write(wine.columns)
    input_features=sel_col.text_input('which feature should be used as the input features','citric acid')

    if n_estimators == 'No limit' :
        regr = RandomForestRegressor(max_depth = max_depth)
    else :
        regr = RandomForestRegressor(max_depth = max_depth,n_estimators=n_estimators)
    X = wine[[input_features]]
    y = wine[['density']]

    regr.fit(X,y)
    
    prediction=regr.predict(X)

    disp_col.subheader('Mean absulute error of the model is :')
    disp_col.write(mean_absolute_error(y,prediction))    
    disp_col.subheader('R squared error of the model is :')
    disp_col.write(r2_score(y,prediction))
