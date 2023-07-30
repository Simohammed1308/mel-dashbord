
#Le lien de l'application dashboard sur stramlit : https://bouchramek-mb-p7-dashboard--p07-dashboard-naj2yx.streamlitapp.com/ 

#!/usr/bin/env python
# coding: utf-8

# In[1]:



import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import math
from urllib.request import urlopen
import json
import requests
import plotly.graph_objects as go 
import shap
import joblib

import warnings
warnings.filterwarnings("ignore")

    
df = pd.read_pickle("df.gz")
print('df shape = ', df.shape)
        
#description des features
        
description = pd.read_csv('HomeCredit_columns_description.csv', 
                                      usecols=['Row', 'Description'], \
                                  index_col=0, encoding='unicode_escape')

@st.cache_data

def get_client_info(data, id_client):
    client_info = data[data['SK_ID_CURR']==int(id_client)]
    return client_info

@st.cache_resource

def plot_distribution(applicationDF,feature, client_feature_val, title):

    if (not (math.isnan(client_feature_val))):
        fig = plt.figure(figsize = (10, 4))

        t0 = applicationDF.loc[applicationDF['TARGET'] == 0]
        t1 = applicationDF.loc[applicationDF['TARGET'] == 1]

        if (feature == "DAYS_BIRTH"):
            sns.kdeplot((t0[feature]/-365).dropna(), label = 'Remboursé', color='g')
            sns.kdeplot((t1[feature]/-365).dropna(), label = 'Défaillant', color='r')
            plt.axvline(float(client_feature_val/-365),  color="blue", linestyle='--', label = 'Position Client')

        elif (feature == "DAYS_EMPLOYED"):
            sns.kdeplot((t0[feature]/365).dropna(), label = 'Remboursé', color='g')
            sns.kdeplot((t1[feature]/365).dropna(), label = 'Défaillant', color='r')    
            plt.axvline(float(client_feature_val/365), color="blue", linestyle='--', label = 'Position Client')

        else:    
            sns.kdeplot(t0[feature].dropna(), label = 'Remboursé', color='g')
            sns.kdeplot(t1[feature].dropna(), label = 'Défaillant', color='r')
            plt.axvline(float(client_feature_val), color="blue",linestyle='--', label = 'Position Client')


        plt.title(title, fontsize='20', fontweight='bold')
            #plt.ylabel('Nombre de clients')
            #plt.xlabel(fontsize='14')
        plt.legend()
        plt.show()  
        st.pyplot(fig)
    else:
        st.write("Comparaison impossible car la valeur de cette variable n'est pas renseignée (NaN)")
        
@st.cache_resource

def univariate_categorical(applicationDF,feature,client_feature_val,titre,ylog=False,label_rotation=False,
                               horizontal_layout=True):
        if (client_feature_val.iloc[0] != np.nan):

            temp = applicationDF[feature].value_counts()
            df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})

            categories = applicationDF[feature].unique()
            categories = list(categories)

            # Calculate the percentage of target=1 per category value
            
            cat_perc = applicationDF[[feature,'TARGET']].groupby([feature],as_index=False).mean()
            cat_perc["TARGET"] = cat_perc["TARGET"]*100
            cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)

            if(horizontal_layout):
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            else:
                fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20,24))

            # 1. Subplot 1: Count plot of categorical column
            # sns.set_palette("Set2")
            s = sns.countplot(ax=ax1, 
                            x = feature, 
                            data=applicationDF,
                            hue ="TARGET",
                            order=cat_perc[feature],
                            palette=['g','r'])

            pos1 = cat_perc[feature].tolist().index(client_feature_val.iloc[0])
            #st.write(client_feature_val.iloc[0])

            # Define common styling
            ax1.set(ylabel = "Nombre de clients")
            ax1.set_title(titre, fontdict={'fontsize' : 15, 'fontweight' : 'bold'})   
            ax1.axvline(int(pos1), color="blue", linestyle='--', label = 'Position Client')
            ax1.legend(['Position Client','Remboursé','Défaillant' ])

            # If the plot is not readable, use the log scale.
            if ylog:
                ax1.set_yscale('log')
                ax1.set_ylabel("Count (log)",fontdict={'fontsize' : 15,'fontweight' : 'bold'})   
            if(label_rotation):
                s.set_xticklabels(s.get_xticklabels(),rotation=90)

            # 2. Subplot 2: Percentage of defaulters within the categorical column
            s = sns.barplot(ax=ax2, 
                            x = feature, 
                            y='TARGET', 
                            order=cat_perc[feature], 
                            data=cat_perc,
                            palette='Set2')

            pos2 = cat_perc[feature].tolist().index(client_feature_val.iloc[0])
            #st.write(pos2)

            if(label_rotation):
                s.set_xticklabels(s.get_xticklabels(),rotation=90)
            plt.ylabel('Pourcentage de défaillants [%]', fontsize=10)
            plt.tick_params(axis='both', which='major', labelsize=10)
            ax2.set_title(titre+" (% Défaillants)",fontdict={'fontsize' : 15, 'fontweight' : 'bold'})
            ax2.axvline(int(pos2), color="blue", linestyle='--', label = 'Position Client')
            ax2.legend()
            plt.show()
            st.pyplot(fig)
        else:
            st.write("Comparaison impossible car la valeur de cette variable n'est pas renseignée (NaN)")
            
            
#Chargement des données    
#df, data_test, data_train, description = load_data()

ignore_features = ['Unnamed: 0','SK_ID_CURR', 'INDEX', 'TARGET']
relevant_features = [col for col in df if col not in ignore_features]

#Chargement du modèle
model =joblib.load("model.joblib")


 #######################################
    # SIDEBAR
#######################################

SHAP_GENERAL = "lgbm_importances01.png"

with st.sidebar:
    st.header(" Prêt à dépenser")

    st.write("## ID Client")
    id_list = df["SK_ID_CURR"].values
    id_client = st.selectbox(
            "Sélectionner l'identifiant du client", id_list)

    st.write("## Actions à effectuer")
    show_credit_decision = st.checkbox("Afficher la décision de crédit")
    show_client_details = st.checkbox("Afficher les informations du client")
    show_client_comparison = st.checkbox("Comparer aux autres clients")
    shap_general = st.checkbox("Afficher la feature importance globale")
    if(st.checkbox("Aide description des features")):
        list_features = description.index.to_list()
        list_features = list(dict.fromkeys(list_features))
        feature = st.selectbox('Sélectionner une variable',  sorted(list_features))
        desc = description['Description'].loc[description.index == feature][:1]
        st.markdown('**{}**'.format(desc.iloc[0]))
        
 #######################################
    # HOME PAGE - MAIN CONTENT
    #######################################

#Titre principal

html_temp = """
    <div style="background-color: gray; padding:10px; border-radius:10px">
    <h1 style="color: white; text-align:center">Dashboard de Scoring Crédit</h1>
    </div>
    <p style="font-size: 20px; font-weight: bold; text-align:center">
    Support de décision crédit à destination des gestionnaires de la relation client</p>
    """
st.markdown(html_temp, unsafe_allow_html=True)

with st.expander(" A quoi sert cette application ?"):
    st.write("Ce dashboard interactif à destination des gestionnaires de la relation client de l'entreprise **Prêt à dépenser** permet de comprendre et interpréter les décisions potentielles (prédictions faites par un modèle d'apprentissage) d'ottroi ou non de crédit aux clients") 
    st.text('\n') 
    st.write("**Objectif**:  répondre au soucis de transparence vis-à-vis des décisions d’octroi de crédit qui va tout à fait dans le sens des valeurs que l’entreprise veut incarner")
    


#Afficher l'ID Client sélectionné
st.write("ID Client Sélectionné :", id_client)
#id_client = st.number_input(label='Veuillez saisir l\'identifiant du client demandeur de crédit:',format="%i", value=0)
if (int(id_client) in id_list):
    client_info = df[df['SK_ID_CURR']==int(id_client)]
    
      #-------------------------------------------------------
        # Afficher la décision de crédit
        #-------------------------------------------------------

    if (show_credit_decision):
        st.header('‍ Scoring et décision du modèle')

            #Appel de l'API :--port 1234 

        #API_url = "http://127.0.0.1:1234/credit/" + str(id_client)
        API_url = "https://api-flask-mel-593f2a859e1d.herokuapp.com/credit/" + str(id_client)
        #st.write(API_url)
        
        
        
        

        with st.spinner('Chargement du score du client...'):
            
            #json_url = urlopen(API_url)
            print(API_url)
            json_url = requests.get(API_url)
            
            #st.write(json_url.json())
            print(json_url)

            API_data = json_url.json()
            classe_predite = API_data['prediction']
            if classe_predite == 1:
                decision = ' Mauvais prospect (Crédit Refusé)'
            else:
                decision = ' Bon prospect (Crédit Accordé)'
            proba = 1-API_data['proba']

            client_score = round(proba*100, 2)

            left_column, right_column = st.columns((1, 2))

            left_column.markdown('Risque de défaut: **{}%**'.format(str(client_score)))
          

            if classe_predite == 1:
                left_column.markdown('Décision: <span style="color:red">**{}**</span>'.format(decision), unsafe_allow_html=True)   
            else:    
                left_column.markdown('Décision: <span style="color:green">**{}**</span>'.format(decision),unsafe_allow_html=True)

            gauge = go.Figure(go.Indicator(
                mode = "gauge+delta+number",
                title = {'text': 'Pourcentage de risque de défaut'},
                value = client_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {'axis': {'range': [None, 100]},
                             'steps' : [
                                 {'range': [0, 25], 'color': "lightgreen"},
                                 {'range': [25, 50], 'color': "lightyellow"},
                                 {'range': [50, 75], 'color': "orange"},
                                 {'range': [75, 100], 'color': "red"},
                                 ],
                             'threshold': {
                            'line': {'color': "black", 'width': 10},
                            'thickness': 0.8,
                            'value': client_score},

                             'bar': {'color': "black", 'thickness' : 0.2},
                            },
                    ))

            gauge.update_layout(width=450, height=250, 
                                    margin=dict(l=50, r=50, b=0, t=0, pad=4))

            right_column.plotly_chart(gauge)

        show_local_feature_importance = st.checkbox(
                "Afficher les variables ayant le plus contribué à la décision du modèle ?")
        if (show_local_feature_importance):
            shap.initjs()
            number = st.slider('Sélectionner le nombre de feautures à afficher ?',2, 20, 8)

            X = df[df['SK_ID_CURR']==int(id_client)]
            X = X[relevant_features]

            fig, ax = plt.subplots(figsize=(15, 15))
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            shap.summary_plot(shap_values[0], X, plot_type ="bar",max_display=number, color_bar=False, plot_size=(8, 8))


            st.pyplot(fig)

         #-------------------------------------------------------
        # Afficher les informations du client
        #-------------------------------------------------------

    personal_info_cols = {
            'CODE_GENDER': "GENRE",
            'DAYS_BIRTH': "AGE",

            'CNT_CHILDREN': "NB ENFANTS",
            'FLAG_OWN_CAR': "POSSESSION VEHICULE",
            'FLAG_OWN_REALTY': "POSSESSION BIEN IMMOBILIER",


            'DAYS_EMPLOYED': "NB ANNEES EMPLOI",
            'AMT_INCOME_TOTAL': "REVENUS",
            'AMT_CREDIT': "MONTANT CREDIT", 

            'AMT_ANNUITY': "MONTANT ANNUITES",


            'EXT_SOURCE_2': "EXT_SOURCE_2",
            'EXT_SOURCE_3': "EXT_SOURCE_3",

        }

    default_list= ["GENRE","AGE","NB ENFANTS","REVENUS","MONTANT CREDIT"]
    numerical_features = [ 'DAYS_BIRTH' ,'CNT_CHILDREN','DAYS_EMPLOYED','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','EXT_SOURCE_1', 
                         'EXT_SOURCE_2 ' , 'EXT_SOURCE_3' ]

    rotate_label = ["NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE"]
    horizontal_layout = ["OCCUPATION_TYPE", "NAME_INCOME_TYPE"]

    if (show_client_details):
        st.header('‍ Informations relatives au client')

        with st.spinner('Chargement des informations relatives au client...'):
            personal_info_df = client_info[list(personal_info_cols.keys())]
               
            personal_info_df.rename(columns=personal_info_cols, inplace=True)

            #personal_info_df["AGE"] = int(round(personal_info_df["AGE"]/365*(-1)))
            #personal_info_df["NB ANNEES EMPLOI"] = int(round(personal_info_df["NB ANNEES EMPLOI"]/365*(-1)))


            filtered = st.multiselect("Choisir les informations à afficher",options =default_list,#list(personal_info_df.columns),
                                      default=list(default_list))
            df_info = personal_info_df[filtered] 
            df_info['SK_ID_CURR'] = client_info['SK_ID_CURR']
            df_info = df_info.set_index('SK_ID_CURR')

            st.table(df_info.astype(str).T)
            show_all_info = st.checkbox("Afficher toutes les informations (dataframe brute)")
            if (show_all_info):
                st.dataframe(client_info)

#-------------------------------------------------------
        # Comparer le client sélectionné à d'autres clients
        #-------------------------------------------------------
#

    if (show_client_comparison):
        st.header('‍ Comparaison aux autres clients')
            #st.subheader("Comparaison avec l'ensemble des clients")
        with st.expander(" Explication de la comparaison faite"):
            st.write("Lorsqu'une variable est sélectionnée, un graphique montrant la distribution de cette variable selon la classe (remboursé ou défaillant) sur l'ensemble des clients (dont on connait l'état de remboursement de crédit) est affiché avec une matérialisation du positionnement du client actuel.") 

        with st.spinner('Chargement de la comparaison liée à la variable sélectionnée'):
            var = st.selectbox("Sélectionner une variable",list(personal_info_cols.values()))
            feature = list(personal_info_cols.keys())[list(personal_info_cols.values()).index(var)]
            
            if (feature in numerical_features):
                
                plot_distribution(data_train, feature, client_info[feature], var)  
                
            elif (feature in rotate_label):
                
                univariate_categorical(df, feature,client_info[feature], var, False, True)
            elif (feature in horizontal_layout):
               
                
                univariate_categorical(df, feature,client_info[feature], var, False, True, True)
                
            else:
                
                univariate_categorical(df, feature, client_info[feature], var)
                                    
       
            
#-------------------------------------------------------
        # Afficher la feature importance globale
        #-------------------------------------------------------

    if (shap_general):
        st.header('‍Feature importance globale')
        st.image('lgbm_importances01.png')
#else:
    #st.markdown("**Identifiant non reconnu**")
                    





