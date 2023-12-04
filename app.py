import os
import utils as utl
from PIL import Image
import time
import streamlit as st
import pandas as pd
import pickle
from prediction_function.module import ciblage, chiffre_affaire,collect_parametres

def main():    
    # Settings
    st.set_page_config(layout="wide", page_title='Demo item')
    utl.set_page_title('Costumer Segmentation')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Loading CSS
    utl.local_css("css/style.css")
    utl.remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')
    # Logo
    dir_root = os.path.dirname(os.path.abspath(__file__))
    logo = Image.open(dir_root+'/logo-lm.png')
    # Selecting a job
    # st.sidebar.image(logo)
    st.sidebar.selectbox('Select',('loren','Ipsum'))
    st.sidebar.multiselect('Multi',('loren','Ipsum'))
    st.sidebar.date_input('Date')
    st.sidebar.text_input('Text')
    st.sidebar.slider('Slider',min_value=5,max_value=20)
    st.warning('Warning')
    st.info('Info')
    st.error('Error')
    
    with utl.stNotification('Sample notification, always on top and floats (spiner is optional)'):
        time.sleep(5)


if __name__ == '__main__':
    main()

st.write(
    """
    ### Costumer Segmentation:
    #### Application de deploiment de modèle de machine learning
    #### basé sur le concept de segmentation du marché.
    Autheur: WORA SOUAMY Louis Martin (@lmlouis) copyright 2022.
    
    Github Repository : https://github.com/lmlouis/customer-segmentation
    
    Notebook Modele :https://github.com/lmlouis/IntroductionIA/blob/main/Customer_Segmentation.ipynb
"""
)
st.title('Fonctionalités Principales')
st.write('''
        * **Cibler le marché** le plus remptable selon le la localisation du client lors de la transaction
        * **Prédir le chiffre d'affaire** d'une quantité de Transaction d'un produit effectué par un client 
         ''')




st.sidebar.header('''Paramètres :''')
df = collect_parametres(st, pd)






st.subheader('1 - Cibler le marché selon la situation géographique du client')
st.write('''
         ### Ciblage
         Prédiction par localisation (latitude et longitude) du lieu de transaction
         valeurs de prediction possibles:
         * Marché Elevé
         * Marché Moyen
         * Marché Faible 
         ''')

st.write(df[['Latitude', 'Longitude']])


ciblage(df, st, pd, pickle)
    

st.subheader("2 - Prédire le chiffre d'affaire d'une transaction pour une quantité de produit acheté")
st.write('''
         ### Prédiction
         Prédiction du chiffre d'affaire:
         ''')

st.write(df['Quantité'])


chiffre_affaire(pickle, st, df)

