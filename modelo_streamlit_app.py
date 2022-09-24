import pandas as pd
import numpy as np
from PIL import Image
from joblib import dump,load
import streamlit as st
from sklearn import *

st.set_page_config(page_title="streamlit", layout="wide")

primaryColor="#F63366"
backgroundColor="#b5b5b5"
texColor="#F0F2F6"
font="sans serif"

st.markdown("<h1 style='text-align: center; color:black;'> ML model deployment </h1>", unsafe_allow_html=True)
st.markdown("<br></br>",unsafe_allow_html=True)

image=Image.open('iris.jpg')
st.image(image)

with st.sidebar:
    clasificador =st.selectbox(label='clasificador', options=('Árbol de decisión', 'K vecinos', 'svc', 'hgbc'))
    st.markdown("<br></br>",unsafe_allow_html=True)

    sepal_length=st.slider(label='sepal length (cm):',min_value=2.0,max_value=9.0,step=0.05)
    sepal_width=st.slider(label='sepal length (cm):',min_value=1.0,max_value=6.0,step=0.05)
    petal_length=st.slider(label='petal length (cm):',min_value=0.0,max_value=8.0,step=0.05)
    petal_width=st.slider(label='petal width (cm):',min_value=0.0,max_value=8.0,step=0.05)
    
def classifier(clasificador):
    if clasificador=='Árbol de decisión':
        tree_classifier=load('clf_dtc_model.py')
        resultado=tree_classifier.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    elif clasificador=='K vecinos':
        knn_classifier=load('clf_knn_model.py')
        resultado=knn_classifier.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    elif clasificador=='svc':
        knn_classifier=load('clf_svc_model.py')
        resultado=knn_classifier.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    elif clasificador=='hgbc':
        knn_classifier=load('clf_hgbc_model.py')
        resultado=knn_classifier.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    else:
        resultado=None
    return resultado

col1,col2,col3=st.columns(3)

flecha=Image.open('flecha.png')

if st.button('clasifique por favor'):
    clase=int(classifier(clasificador))
    if clase==0:
        col1.image(flecha)
    elif clase==1:
        col2.image(flecha)
    elif clase==2:
        col2.image(flecha)
    else:
        pass
else:
    st.write('elija clasificador y parámetros')
