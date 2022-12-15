

def ciblage(df, st, pd, pickle):
    lieu =pd.DataFrame()
    lieu['lat'] = df['Latitude']
    lieu['lon'] = df['Longitude']
    st.map(lieu, zoom=3)

    kmeans_modele = pickle.load(open('model_pikle/ciblage_par_Latitude_et_Longitude.pkl','rb'))

    st.subheader('Prédiction')
    value = kmeans_modele.predict(df[['Latitude', 'Longitude']])
    if (value[0] == 0):
        st.write("## Ce lieu appartient au marché élévé")

    elif (value[0] == 1):
        st.write("## Ce lieu appartient au marché moyen")

    elif (value[0] == 2):
        st.write("## Ce lieu appartient au marché faible")

def chiffre_affaire(pickle, st, df):
    LR_modele = pickle.load(open('model_pikle/prediction_CFA_par_stock_vendu.pkl','rb'))

    CFA = LR_modele.predict([df['Quantité']])
    st.metric(label="Chiffre d'Affaire", value=f"{round(CFA[0],2)} £.", delta=f"soit {round(CFA[0]/df['Quantité'].iloc[0],2)} £ l'unité")
    

def collect_parametres(st, pd):
    Latitude = st.sidebar.slider("Latitude :",-90.0,90.0,15.0)
    Longitude = st.sidebar.slider("Longitude :",-180.0,180.0,15.0)
    Quantite = st.sidebar.slider("Quantité :", 0,20, 4)
    parametres = {
        'Latitude': Latitude,
        'Longitude': Longitude,
        'Quantité': Quantite
    }
    
    st.sidebar.write("Latitude (°) : Latitude du Lieu de la Transaction")
    st.sidebar.write("Longitude (°): Longitude du Lieu de la Transaction")
    st.sidebar.write("Quantité (entier): Quantité du Produit achété lors de la Transation")
    dataframe_parametres = pd.DataFrame(parametres, index=[0])
    
    return dataframe_parametres