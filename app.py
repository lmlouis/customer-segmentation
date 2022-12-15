import streamlit as st
import pandas as pd
import pickle
from prediction_function.module import ciblage, chiffre_affaire,collect_parametres

with open("css/style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

 
st.title('costumer Segmention')
st.write('''
         ### Application de deploiment de modèle de machine learning
         ### basé sur le concept de segmentation du marché. 
         
        ## Fonctionalités Principales 
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

