################
# vigitel-analysis
#################
# Fabiano Novaes Barcellos Filho 2019
# 30/11/19
###################################################

# to run: streamlit run hello.py

import streamlit as st
import pandas as pd
import numpy as np 
##import pandas_profiling as pf

#datapackage
# from datapackage import Package

#preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

#models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

#metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

#explainable models
#import shap
# from interpret.glassbox import ExplainableBoostingClassifier

st.title('Algorítmo Vigitel Predição de Diabetes')

########## READ DATA ##########
FILENAME = '/Users/fabianofilho/Projects/vigitel/data/all2006.csv'
NAN=['555', '777', '888',555, 777, 888,'NaN', 0, 'nan', 'na']
dtypes = {
    #Variable name	Type
    'chave':	np.object,
    'replica':	np.float64,
    'ano':	np.float64,
    'mesfim': 	np.object,
    'cidade':	np.object,
    'regiao':	np.object,
    'bairro':	np.object,
    'hora_ini':	np.float64,
    'hora_fim':	np.float64,
    'duracao':	np.object,
    'operador':	np.object,
    'operadora':	np.object,
    'q6':	np.float64,
    'q7':	np.object,
    'civil':	np.object,
    'q8a':	np.object,
    'q8b':	np.object,
    'q8_anos':	np.object,
    'r128a':	np.object,
    'q9':	np.object,
    'q11':	np.object,
    'q14':	np.object,
    'q15':	np.object,
    'q15a':	np.object,
    'q16':	np.object,
    'q16a':	np.object,
    'q17':	np.object,
    'q18':	np.object,
    'q19':	np.object,
    'q20':	np.object,
    'q21':	np.object,
    'q23':	np.object,
    'q25':	np.object,
    'q26':	np.object,
    'q27':	np.object,
    'q28':	np.object,
    'q28a':	np.object,
    'q29':	np.object,
    'q29a':	np.object,
    'q30a':	np.object,
    'q31a':	np.object,
    'r143a': np.object,
    'r171':	np.object,
    'r172':	np.object,
    'r173':	np.object,
    'r144a':	np.object,
    'r144b':	np.object,
    'q35':	np.object,
    'q36':	np.object,
    'q37':	np.object,
    'q38':	np.object,
    'q39':	np.object,
    'r200':	np.float64,
    'q40':	np.object,
    'q40b':	np.object,
    'q42':	np.object,
    'q43a':	np.object,
    'q44':	np.object,
    'q45':	np.object,
    'q46':	np.object,
    'q47':	np.object,
    'q48':	np.object,
    'q49':	np.object,
    'r147':	np.object,
    'r148_hh':	np.float64,
    'r148_mm':	np.float64,
    'q50':	np.object,
    'q51':	np.object,
    'q52':	np.object,
    'q53':	np.object,
    'q54':	np.object,
    'q55':	np.object,
    'q56':	np.object,
    'r149':	np.object,
    'r150_hh':	np.float64,
    'r150_mm':	np.float64, 
    'q59a':	np.object,
    'q59b':	np.object,
    'q59c':	np.object,
    'q60':	np.object,
    'q61':	np.float64,
    'q61a':	np.float64,
    'q61_fx':	np.object,
    'q61a_fx':	np.object,
    'q62':	np.float64,
    'q63':	np.object,
    'q64':	np.object,
    'q67':	np.object,
    'q68':	np.object,
    'r157':	np.object,
    'q69':	np.object,
    'q69_ou':	np.object,
    'q70':	np.object,
    'q71':	np.object,
    'q74':	np.object,
    'q75':	np.object,
    'r203':	np.object,
    'r129':	np.object,
    'r130a':	np.object,
    'r174':	np.object,
    'q76':	np.object,
    'r138':	np.object,
    'r202':	np.object,
    'r204':	np.object,
    'r133a':	np.object,
    'r134c':	np.object,
    'd3':	np.object,
    'r133b':	np.object,
    'r134b':	np.object,
    'd1':	np.object,
    'q79a':	np.object,
    'q80':	np.object,
    'q81':	np.object,
    'q82':	np.object,
    'q88':	np.object,
    'r135':	np.object,
    'r136':	np.object,
    'r153':	np.object,
    'r137a':np.object,
    'r154':	np.object,
    'r155':	np.object,
    'r156':	np.object,
    'r178':	np.object,
    'r179':	np.object,
    'r180':	np.float64,
    'r900':	np.object,
    'obs_r900':	np.object,
    'moradores':	np.object,
    'adultos':	np.object,
    'obs':	np.object,
    'fet':	np.object,
    'cat_esc':	np.object,
    'fesc':	np.object,
    'fxesc': 	np.object,
    'pesorake ':	np.float64,
    'q9_i':	np.float64,
    'q11_i':	np.float64,
    'pinterno':	np.float64,
    'q59a_horas':	np.float64,
    'q59c_horas':	np.float64,
    # aqui começa a repetir os campos
    'fumante': np.object,	
    'exfuma':	np.object,
    'mais20':	np.object,
    'fumocasa':	np.object,
    'fumotrab':	np.object,
    'imc':	np.float,
    'imc_i':	np.float,
    'excpeso':	np.object,
    'excpeso_i':	np.object,
    'obesid':	np.object,
    'obesid_i':	np.object,
    'hortareg':	np.object,
    'frutareg':	np.object,
    'flvreg':	np.object,
    'cruadia':	np.float,
    'cozidadia':	np.float,
    'hortadia':	np.float,
    'sucodia':	np.float,
    'sofrutadia':	np.float,
    'frutadia':	np.float,
    'flvdia':	np.float,
    'flvreco':	np.object,
    'refritl5':	np.object,
    'feijao5':	np.object,
    'lanche_7':	np.object,
    'af': 	np.object,
    'freq':	np.object,
    'time': 	np.object,
    'ati_livre': 	np.object,
    'ativo_livre':	np.object,
    'atitrans':	np.object,
    'atidom':	np.object,
    'atiocu':	np.object,
    'inativo':	np.object,
    'q51medio':	np.float,
    'q54medio':	np.float ,
    'deslocdia': 	np.float ,
    'deslocsemana': 	np.float, 
    'atiocusemana': 	np.float ,
    'faxinasemana': 	np.float ,
    'af3dominios':	np.object,
    'af3dominios_insu':	np.object,
    'tv_d_3':	np.object,
    'tempo_tela_stv':	np.object,
    'tempo_tela_total':	np.object,
    'alcabu':	np.object,
    'direcao':	np.object,
    'direcao_alc':	np.object,
    'saruim':	np.object,
    'iddmamo':	np.object,
    'mamo':	np.object,
    'mamodois':	np.object,
    'iddpapa_old':	np.object,	
    'iddpapa':	np.float,
    'papa':	np.object,
    'papatres':	np.object,
    'hart':	np.object,
    'diab':	np.object,
    'has':	np.object,
    'ind_med_has':	np.object,
    'med_has':	np.object,
    'trat_med_has': np.object,	
    'db':	np.object,
    'ind_med_db': 	np.object,	
    'med_db':	np.object,
    'insulina':	np.object,
    'trat_med_db':	np.object
    }

@st.cache(suppress_st_warning=True)
def loadData():
    dfi_raw = pd.read_csv(FILENAME, dtype=dtypes, na_values=NAN)
    data = pd.read_csv('data/all.csv', dtype=dtypes,na_values=NAN)
    df1 = pd.read_csv('data/all2007.csv', dtype=dtypes,na_values=NAN)
    df2 = pd.read_csv('data/all2008.csv', dtype=dtypes,na_values=NAN)
    df3 = pd.read_csv('data/all2009.csv', dtype=dtypes,na_values=NAN)
    df4 = pd.read_csv('data/all2010.csv', dtype=dtypes,na_values=NAN)
    df5 = pd.read_csv('data/all2011.csv', dtype=dtypes,na_values=NAN)
    df6 = pd.read_csv('data/all2012.csv', dtype=dtypes,na_values=NAN)
    df7 = pd.read_csv('data/all2013.csv', dtype=dtypes,na_values=NAN)
    df8 = pd.read_csv('data/all2014.csv', dtype=dtypes,na_values=NAN)
    df9 = pd.read_csv('data/all2015.csv', dtype=dtypes,na_values=NAN)
    df10 = pd.read_csv('data/all2016.csv', dtype=dtypes, na_values=NAN)
    df11 = pd.read_csv('data/all2017.csv', dtype=dtypes,na_values=NAN)
    return dfi_raw, data, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11

dfi_raw, data, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11 = loadData()

########## SHOW YOUR DATA #########
if st.checkbox('Dados Vigitel'):
	st.subheader("Sistema de Vigilância de Fatores de Risco para doenças crônicas não transmissíveis (DCNT) do Ministério da Saúde")
	st.text('Download: http://svs.aids.gov.br/download/Vigitel/')
	st.write(dfi_raw.head())

@st.cache
# Basic and common preprocessing required for all the models.
def preprocessing(dfi_raw, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11):

    #drop repetitive cols
    cols = ['fumante',
    'mais20',
    'iddfuma',
    'exfuma',
    'iddexfu',
    #  'fumapass',
    'imc',
    'excpeso',
    'obesid',
    'hortareg',
    'frutareg',
    'flvreg',
    'cruadia',
    'cozidadia',
    'hortadia',
    #  'sucodia',
    'sofrutadia',
    'frutadia',
    'flvdia',
    'flvreco',
    'carneg',
    'franpl',
    'gordura',
    'leiteint',
    'refri5',
    'atilaz',
    'atiocu',
    'atitrans',
    'atidom',
    'inativo',
    'alcabu',
    #  'direcao',
    'saruim',
    #  'iddmamo',
    #  'mamo',
    #  'mamodois',
    #  'iddpapa',
    #  'papa',
    #  'papatres',
    #  'protuv',
    'hart',
    'diab',
    'coracao',
    'dislipi',
    'osteo',
    #  'asma',
    #  'plano',
    #  'planeja',
    #  'iddplfa',
    #  'metofa',
    #  'samtl',
    #  'samat',
    'sobrepeso',
    'obesidmorb',
    'feijao5',
    'atilaz_trans',
    'tv_d3',
    'idoso_bp',
    'idoso_pa',
    'idoso_sp',
    'pesorake']
    dfi = dfi_raw.drop(cols,1)

    #preencher os faltantes com 0 (valor que outras features nao tem)
    dfi = dfi.fillna(0)
    
    #separar quem nao tem diabetes inicialmente para predizer quem virá a ter
    dfsd = dfi[dfi['q76'] == '2.0']
    
    #juntar os dataframes pela ordem
    dfv1 = pd.merge(dfsd, df1, on='ordem', how='left', suffixes=['_0','_1'])
    dfv2 = pd.merge(dfsd, df2, on='ordem', how='left', suffixes=['_0','_2'])
    dfv3 = pd.merge(dfsd, df3, on='ordem', how='left', suffixes=['_0','_3'])
    dfv4 = pd.merge(dfsd, df4, on='ordem', how='left', suffixes=['_0','_4'])
    dfv5 = pd.merge(dfsd, df5, on='ordem', how='left', suffixes=['_0','_5'])
    dfv6 = pd.merge(dfsd, df6, on='ordem', how='left', suffixes=['_0','_6'])
    dfv7 = pd.merge(dfsd, df7, on='ordem', how='left', suffixes=['_0','_7'])
    dfv8 = pd.merge(dfsd, df8, on='ordem', how='left', suffixes=['_0','_8'])
    
    dfv = pd.concat([dfv1,dfv2,dfv3,dfv4,dfv5,dfv6,dfv7,dfv8,
    #                dfv9,dfv10,dfv11
                    ])
    
    #dando classe ao diabetes
    dfv['class'] = 0

    # 1 ano, 5 anos e 10 anos de predição
    ## 1ano
    dfv_1a = dfv
    dfv_1a['class'] = 'Não doente'
    dfv_1a.loc[(dfv_1a['q76_1'] == '1.0'), 'class'] = 'Ficou doente'
    dfv_1a['class'].value_counts()

    ## 5anos
    dfv_5a = dfv
    dfv_5a['class'] = 'Não doente'
    dfv_5a.loc[(dfv_5a['q76_1'] == '1.0') | \
            (dfv_5a['q76_2'] == '1.0') | \
            (dfv_5a['q76_3'] == '1.0') | \
            (dfv_5a['q76_4'] == '1.0') | \
            (dfv_5a['q76_5'] == '1.0'), 'class'] = 'Ficou doente'
    dfv_5a['class'].value_counts()

    ## 10 anos
    dfv_10a = dfv
    dfv_10a['class_0'] = 'Não doente'
    dfv_10a.loc[(dfv_10a['q76_1'] == '1.0') | \
                (dfv_10a['q76_2'] == '1.0') | \
                (dfv_10a['q76_3'] == '1.0') | \
                (dfv_10a['q76_4'] == '1.0') | \
                (dfv_10a['q76_5'] == '1.0') | \
                (dfv_10a['q76_6'] == '1.0') | \
                (dfv_10a['q76_7'] == '1.0') 
                # | \
                # (dfv_10a['q76_8'] == '1.0') 
    #             | \
    #             (dfv_10a['q76_9'] == '1.0') | \
    #             (dfv_10a['q76_10'] == '1.0') | \
    #             (dfv_10a['q76_11'] == '1.0')
                , 'class_0'] = 'Ficou doente'
    # dfv_10a['class_0'].value_counts()

    #### regex future data for final dataframe
    dff = dfv_10a.filter(regex=r'_0', axis=1)

    #dummies
    # 'q80', 'q52', 'q88', 'q43a', 'r144a', 'r149', 'r134c', 'r144b', 'q23', 'r137a', 
    # 'r134b', 'q82', 'r147', 'fxesc', 'q39', 'q59c', 'q61_fx', 'q61a_fx', 'q16', 
    # 'q31a', 'q40b', 'q30a', 'q25', 'q53', 'q54', 'r136', 'q26', 'q21', 'r130a', 'q59a'

    dummy_cols=['cidade_0', 'regiao_0', 'civil_0', 'q8a_0', 'q15_0', 'q17_0',
                'q18_0', 'q19_0', 'q20_0', 'q27_0', 
                'q28_0', 'q29_0', 'q36_0', 
                'q44_0', 'q45_0', 'q46_0', 'q47_0', 
                'q50_0', 'q51_0', 'q55_0', 'q56_0', 
                'q60_0', 'q69_0', 'q71_0', 
                'q74_0', 'fet_0', 'fesc_0']

    dff_dm = pd.get_dummies(dff, columns=dummy_cols)

    #removendo os NaNs que surgiram no merge
    dff_dm.dropna(inplace=True)

    #criando variáveis dependentes e independente
    X = dff_dm.drop('class_0', 1)
    y = dff_dm['class_0'].to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(dff_dm['class_0'])

    #fazendo o balanceamento de dados pela minoria
    smote = SMOTE(ratio='minority')
    X_sm, y_sm = smote.fit_sample(X, y)
   
    #separando treino e teste com 70% treino 30% teste
    X_train, X_test, y_train, y_test = train_test_split(X_sm,y_sm, test_size=0.3,random_state=42)
    return X_sm, y_sm, X_train, X_test, y_train, y_test

X_sm, y_sm, X_train, X_test, y_train, y_test = preprocessing(dfi_raw, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11)


### Accepting user data for predicting its Member Type
# @st.cache(suppress_st_warning=True)
# def accept_user_data():
# 	#MÃE
# 	var1 = st.text_input('Colocar dado:')
# 	user_prediction_data = np.array([var1,var2]).reshape(1,-1)
# 	user_prediction_data = user_prediction_data.astype(float)
# 	return user_prediction_data


# Training Models for Classification
################# TRAINING DT ##################
# Training Decision Tree for Classification.
@st.cache(suppress_st_warning=True)
def decisionTree(X_train, X_test, y_train, y_test):
		# Train the model
		tree = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
		tree.fit(X_train, y_train)
		y_pred = tree.predict(X_test)
		cv_score = cross_val_score(tree, X_sm, y_sm, cv=10)
		score = metrics.accuracy_score(y_test, y_pred) * 100
		report = classification_report(y_test, y_pred)

		return cv_score, score, report, tree

################# TRAINING LR ##################
# Training LogisticRegression for Classification.
@st.cache(suppress_st_warning=True)
def logisticRegression(X_train, X_test, y_train, y_test):
		# Train the model
		lr = LogisticRegression()
		lr.fit(X_train, y_train)
		y_pred = lr.predict(X_test)
		cv_score = cross_val_score(lr, X_sm, y_sm, cv=10)
		score = metrics.accuracy_score(y_test, y_pred) * 100
		report = classification_report(y_test, y_pred)

		return cv_score, score, report, lr

################# TRAINING RF ##################
# Training RandomForest for Classification.
@st.cache(suppress_st_warning=True)
def randomForest(X_train, X_test, y_train, y_test):
		# Train the model
		rf = RandomForestClassifier(random_state=42)
		rf.fit(X_train, y_train)
		y_pred = rf.predict(X_test)
		cv_score = cross_val_score(rf, X_sm, y_sm, cv=10)
		score = metrics.accuracy_score(y_test, y_pred) * 100
		report = classification_report(y_test, y_pred)

		return cv_score, score, report, rf

################# TRAINING XGB ##################
# Training XGBoost for Classification.
@st.cache(suppress_st_warning=True)
def xgBoost(X_train, X_test, y_train, y_test):
		# Train the model
		xg = XGBClassifier(random_state=42)
		xg.fit(X_train, y_train)
		y_pred = xg.predict(X_test)
		cv_score = cross_val_score(xg, X_sm, y_sm, cv=10)
		score = metrics.accuracy_score(y_test, y_pred) * 100
		report = classification_report(y_test, y_pred)

		return cv_score, score, report, xg

################# TRAINING NNT ##################
# Training Neural Network for Classification.
@st.cache(suppress_st_warning=True)
def neuralNet(X_train, X_test, y_train, y_test):
		# Scalling the data before feeding it to the Neural Network.
		scaler = StandardScaler()  
		scaler.fit(X_train)  
		X_train = scaler.transform(X_train)  
		X_test = scaler.transform(X_test)
		# Instantiate the Classifier and fit the model.
		clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		cv_score = cross_val_score(clf, X_sm, y_sm, cv=10)
		score = metrics.accuracy_score(y_test, y_pred) * 100
		report = classification_report(y_test, y_pred)
		
		return cv_score, score, report, clf


# Choose how you want to analyse
choose_analysis = st.sidebar.selectbox("Escolha sua análise",
			   ["Nada por enquanto", "Visualização dos dados", "Seleção de Características", "Predição do Vigitel atual", "Preenchimento de um novo formulário Vigitel", "Explicação dos modelos"])

# Choose your machine learning model
choose_model = st.sidebar.selectbox("Escolha seu modelo",
			   ["Nenhum", "Árvore de Decisão", "Regressão Logística", "Random Forest", 'XGBoost', 'Redes Neurais'])

if choose_analysis == "Nada por enquanto":
		st.write(" ")
		st.write("Podemos começar escolhendo uma forma de análise")

####### VISUALIZAÇÃO #####
# Visualização dos dados
if choose_analysis == 'Visualização dos dados':
	######### CODMUN ##########
	codmun = st.selectbox('Selecionar o Município', data.cidade.unique(),1)
	ano = st.slider('Selecionar o Ano', 2006,2017,0,1)
	data = data[data['cidade']==codmun]
	st.bar_chart(data['class_0'])
	#map??


######## PREDIÇÃO ######
# Predição de todo o banco de dados
if choose_analysis == 'Predição do VIGITEL atual':
	# NONE 
	if choose_model == "Nenhum":
		st.write(" ")
		st.write("A partir dos dados da Vigitel, agora é possível predizer Doenças Crônicas como Diabetes!")
		st.write(" ")
		st.write("Escolha seu modelo!!")

	# SETTING DT 
	elif choose_model == "Árvore de Decisão":
		cv_score, score, report, tree = decisionTree(X_train, X_test, y_train, y_test)
		st.text("A acurácia da Árvode de Decisão é: ")
		st.write(score,"%")
		st.text('O score na validação cruzada foi:')
		st.write(cv_score.mean(), "%")
		st.text("O relatório da Classificação da Árvore de Decisão é: ")
		st.write(report)

	################# SETTING LR ####################
	elif choose_model == "Regressão Logística":
		cv_score, score, report, lr = logisticRegression(X_train, X_test, y_train, y_test)
		st.text("A acurácia da Regressão Logística é: ")
		st.write(score,"%")
		st.text('O score na validação cruzada foi:')
		st.write(cv_score.mean(), "%")
		st.text("O relatório da Classificação da Regressão Logística é: ")
		st.write(report)					

	################# SETTING RF ####################
	elif choose_model == "Random Forest":
		cv_score, score, report, rf = randomForest(X_train, X_test, y_train, y_test)
		st.text("A acurácia da Random Forest é: ")
		st.write(score,"%")
		st.text('O score na validação cruzada foi:')
		st.write(cv_score.mean(), "%")
		st.text("O relatório da Classificação da Random Forest é: ")
		st.write(report)

	################# SETTING XGBoost ####################
	elif choose_model == "XGBoost":
		cv_score, score, report, clf = xgBoost(X_train, X_test, y_train, y_test)
		st.text("A acurácia da XGBoost é: ")
		st.write(score,"%")
		st.text('O score na validação cruzada foi:')
		st.write(cv_score.mean(), "%")
		st.text("O relatório da Classificação da XGBoost é: ")
		st.write(report)

	################# SETTING NNT ####################

	elif choose_model == "Redes Neurais":
		cv_score, score, report, clf = neuralNet(X_train, X_test, y_train, y_test)
		st.text("A acurácia da Rede Neural é: ")
		st.write(score,"%")
		st.text('O score na validação cruzada foi:')
		st.write(cv_score.mean(), "%")
		st.text("O relatório da Classificação da Rede Neural é: ")
		st.write(report)


# Preenchimento de um novo formulário
# elif choose_analysis == 'Preenchimento de um novo formulário Vigitel':
# 	user_prediction_data, user_prediction_data_float = accept_user_data()
# 	################ NONE #############
# 	if choose_model == "Nenhum":
# 		st.write(" ")
# 		st.write("A partir dos dados da Vigitel, agora é possível predizer Doenças Crônicas!")

# 	################ SETTING DT #############
# 	elif choose_model == "Árvore de Decisão":
# 		cv_score, score, report, tree = decisionTree(X_train, X_test, y_train, y_test)
# 		pred = tree.predict(user_prediction_data)
# 		st.text("A classe predita é?")
# 		prob = tree.predict_proba(user_prediction_data_float)
# 		st.text("A probablilidade de diabetes (desfecho = 1) é: ", prob*100, "%")

# 	################# SETTING LR ####################
# 	elif choose_model == "Regressão Logística":
# 		cv_score, score, report, lr = decisionTree(X_train, X_test, y_train, y_test)
# 		pred = lr.predict(user_prediction_data)
# 		st.text("A classe predita é:", pred)
# 		prob = lr.predict_proba(user_prediction_data_float)
# 		st.text("A probablilidade de diabetes (desfecho = 1) é: ", prob*100, "%")
	
# 	################ SETTING RF ####################

# 	elif choose_model == "Random Forest":
# 		cv_score, score, report, rf = randomForest(X_train, X_test, y_train, y_test)
# 		pred = rf.predict(user_prediction_data)
# 		st.text("A classe predita é:", pred)
# 		prob = rf.predict_proba(user_prediction_data_float)
# 		st.text("A probablilidade de diabetes (desfecho = 1) é: ", prob*100, "%")

# 	################# SETTING XGB ####################

# 	elif choose_model == "XGBoost":
# 		cv_score, score, report, xg = xgBoost(X_train, X_test, y_train, y_test)
# 		pred = xg.predict(user_prediction_data)
# 		st.text("A classe predita é:", pred)
# 		prob = xg.predict_proba(user_prediction_data_float)
# 		st.text("A probablilidade de diabetes (desfecho = 1) é: ", prob*100, "%")
	
# 	################# SETTING NNT ####################
# 	elif choose_model == "Redes Neurais":
# 		cv_score, score, report, clf = neuralNet(X_train, X_test, y_train, y_test)
# 		pred = clf.predict(user_prediction_data)
# 		st.text("A classe predita é:", pred)
# 		prob = clf.predict_proba(user_prediction_data_float)
# 		st.text("A probablilidade de diabetes (desfecho = 1) é: ", prob*100, "%")


# Seleção de features 
# if choose_analysis == 'Seleção de features':
# 		################ NONE #############
# 	if choose_model == "Nenhum":
# 		st.write(" ")
# 		st.write("A partir dos modelos preditos, agora conseguimos descobrir as características mais importantes.")
# 		st.write(" ")
# 		st.write("Escolha seu modelo!!")

# 	################ SETTING DT #############
# 	elif choose_model == "Árvore de Decisão":
# 		cv_score, score, report, tree = decisionTree(X_train, X_test, y_train, y_test)
# 		sfs = SFS(tree, 
# 		   k_features=n_features, 
# 		   forward=True, 
# 		   floating=False, 
# 		   verbose=2,
# 		   scoring='accuracy',
# 		   cv=0)
# 		feature_names=X.columns
# 		sf = sfs.fit(X_sm, y_sm, custom_feature_names=feature_names)
# 		sf.subsets_

# 	################# SETTING LR ####################
# 	elif choose_model == "Regressão Logística":
# 		cv_score, score, report, lr = logisticRegression(X_train, X_test, y_train, y_test)
# 		sfs = SFS(lr, 
# 		   k_features=n_features, 
# 		   forward=True, 
# 		   floating=False, 
# 		   verbose=2,
# 		   scoring='accuracy',
# 		   cv=0)					
# 		feature_names=X.columns
# 		sf = sfs.fit(X_sm, y_sm, custom_feature_names=feature_names)
# 		sf.subsets_
		
# 	################# SETTING RF ####################
# 	elif choose_model == "Random Forest":
# 		cv_score, score, report, rf = randomForest(X_train, X_test, y_train, y_test)
# 		sfs = SFS(rf, 
# 		   k_features=n_features, 
# 		   forward=True, 
# 		   floating=False, 
# 		   verbose=2,
# 		   scoring='accuracy',
# 		   cv=0)
# 		feature_names=X.columns
# 		sf = sfs.fit(X_sm, y_sm, custom_feature_names=feature_names)
# 		sf.subsets_

# 	################# SETTING XGBoost ####################
# 	elif choose_model == "XGBoost":
# 		cv_score, score, report, xg = xgBoost(X_train, X_test, y_train, y_test)
# 		sfs = SFS(xg, 
# 		   k_features=n_features, 
# 		   forward=True, 
# 		   floating=False, 
# 		   verbose=2,
# 		   scoring='accuracy',
# 		   cv=0)
# 		feature_names=X.columns
# 		sf = sfs.fit(X_sm, y_sm, custom_feature_names=feature_names)
# 		sf.subsets_

# 	################# SETTING NNT ####################
# 	elif choose_model == "Redes Neurais":
# 		cv_score, score, report, clf = neuralNet(X_train, X_test, y_train, y_test)
# 		sfs = SFS(xg, 
# 		   k_features=n_features, 
# 		   forward=True, 
# 		   floating=False, 
# 		   verbose=2,
# 		   scoring='accuracy',
# 		   cv=0)
# 		feature_names=X.columns
# 		sf = sfs.fit(X_sm, y_sm, custom_feature_names=feature_names)
# 		sf.subsets_

#Explicação dos modelos
#SHAP ou interpretML

#   ################# SHAP ################
# j = 0
# # initialize js for SHAP
# shap.initjs()
# explainer = shap.TreeExplainer(xg)
# shap_values = explainer.shap_values(X_sm)
# shap.summary_plot(shap_values, X_sm)
# shap.summary_plot(shap_values, X_sm, plot_type="bar")

#   ############ INTERPRETML ##############
# ebm = ExplainableBoostingClassifier(feature_names=X.columns)
# ebm.fit(X_train, y_train)
# from interpret import show

# ebm_global = ebm.explain_global()
# show(ebm_global)