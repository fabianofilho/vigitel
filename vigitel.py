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


st.title('Algorítmo blue Predição de Diabetes')

########## READ DATA ##########
FILENAME = '/Users/fabianofilho/Projects/vigitel/data/all.csv'

@st.cache(suppress_st_warning=True)
def loadData():
	data = pd.read_csv(FILENAME, index_col=0)
	return data

data = loadData()

########## SHOW YOUR DATA #########
if st.checkbox('Dados Vigitel'):
	st.subheader("Sistema de Vigilância de Fatores de Risco para doenças crônicas não transmissíveis (DCNT) do Ministério da Saúde")
	st.text('Download: http://svs.aids.gov.br/download/Vigitel/')
	st.write(data.head())

@st.cache
# Basic and common preprocessing required for all the models.
def preprocessing(data):

	 #SEXO, PESO, APGAR5 IDANOMAL, GESTACAO, GRAVIDEZ, PARTO, CONSULTAS, ESTCIVMAE
	dndo = data[['SEXO_dn', 'PESO_dn', 'APGAR5', 'IDANOMAL', 'IDADEMAE_dn','GESTACAO_dn', 'GRAVIDEZ_dn', 'PARTO_dn', 'CONSULTAS', 'ESTCIVMAE', 'MORTALIDADE']]

	dn = dndo[dndo.MORTALIDADE==0]
	do = dndo[dndo.MORTALIDADE==1]

	dndo.dropna(inplace=True)

	dndo['GESTACAO_dn'] = dndo['GESTACAO_dn'].astype(object)
	dndo['CONSULTAS'] = dndo['CONSULTAS'].astype(object)
	dndo['ESTCIVMAE'] = dndo['ESTCIVMAE'].astype(object)
	dndo['SEXO_dn'] = dndo['SEXO_dn'].astype(object)
	dndo['IDANOMAL'] = dndo['IDANOMAL'].astype(object)
	dndo['GRAVIDEZ_dn'] = dndo['GRAVIDEZ_dn'].astype(object)
	dndo['PARTO_dn'] = dndo['PARTO_dn'].astype(object)

	dndo = pd.get_dummies(dndo)

	dndo = shuffle(dndo)

	# Assign X (independent features) and y (dependent feature i.e. df['Member type'] column in dataset)
	X = dndo[['SEXO_dn_F', 'SEXO_dn_M', 'PESO_dn', 'APGAR5', 'IDANOMAL_1.0', 'IDANOMAL_2.0', 'IDANOMAL_9.0', 'IDADEMAE_dn', 'GESTACAO_dn_2.0', 'GESTACAO_dn_3.0', 'GESTACAO_dn_4.0', 'GESTACAO_dn_5.0','GESTACAO_dn_6.0', 'GESTACAO_dn_9.0', 'GRAVIDEZ_dn_1.0', 'GRAVIDEZ_dn_2.0', 'CONSULTAS_1.0', 'CONSULTAS_2.0', 'CONSULTAS_3.0', 'CONSULTAS_4.0', 'CONSULTAS_9.0', 'PARTO_dn_1.0', 'ESTCIVMAE_1.0', 'ESTCIVMAE_2.0', 'ESTCIVMAE_3.0','ESTCIVMAE_4.0', 'ESTCIVMAE_5.0']]
	y = dndo['MORTALIDADE'].to_numpy()
	le = LabelEncoder()
	y = le.fit_transform(dndo['MORTALIDADE'])
	smote = SMOTE(ratio='minority')
	X_sm, y_sm = smote.fit_sample(X, y)

	X_train, X_test, y_train, y_test = train_test_split(X_sm,y_sm, test_size=0.3,random_state=42)
	return X_sm, y_sm, X_train, X_test, y_train, y_test

X_sm, y_sm, X_train, X_test, y_train, y_test = preprocessing(data)


# Accepting user data for predicting its Member Type
@st.cache(suppress_st_warning=True)
def accept_user_data():
	#MÃE
	idademae = st.text_input('Colocar IDADE da MÃE:')
    # idanomal_9 = st.text_input('')

	user_prediction_data = np.array([sexo_M, sexo_F, peso, apgary,idanomal_1, idanomal_2, idanomal_9,idademae, gestacao_2, gestacao_3, gestacao_4, gestacao_5, gestacao_6, gestacao_9,gravidez_1, gravidez_2,parto_1,consultas_1, consultas_2, consultas_3, consultas_4, consultas_9, estcivmae_1, estcivmae_2, estcivmae_3, estcivmae_4, estcivmae_5]).reshape(1,-1)
	user_prediction_data = user_prediction_data.astype(float)

	return user_prediction_data


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
			   ["Nada por enquanto", 'Visualização dos dados', "Predição do SINASC atual", "Preenchimento de um novo formulário DN", 'Explicação dos modelos'])

# Choose your machine learning model
choose_model = st.sidebar.selectbox("Escolha seu modelo",
			   ["Nenhum", "Árvore de Decisão", "Regressão Logística", "Random Forest", 'XGBoost', 'Redes Neurais'])


if choose_analysis == "Nada por enquanto":
		st.write(" ")
		st.write("Podemos começar escolhendo uma forma de análise")




# Visualização dos dados
if choose_analysis == 'Visualização dos dados':
	######### CODMUN ##########
	codmun = st.selectbox('Selecionar o Município', ['Vitória', '']
		data.CODMUNNASC.unique(),1)


	apgar = st.slider('Selecionar o APGAR5', 0,10,0,1)
	data = data[data['CODMUNNASC']==codmun]
	st.bar_chart(data['CONSULTAS'])
	#map??





# Predição de todo o banco de dados

if choose_analysis == 'Predição do SINASC atual':
	################ NONE #############
	if choose_model == "Nenhum":
		st.write(" ")
		st.write("A partir dos dados da Declaração de Nascido vivo, agora é possível predizer a Mortalidade Infantil!")
		st.write(" ")
		st.write("Escolha seu modelo!!")

	################ SETTING DT #############
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
elif choose_analysis == 'Preenchimento de um novo formulário DN':
	user_prediction_data, user_prediction_data_float = accept_user_data()
	################ NONE #############
	if choose_model == "Nenhum":
		st.write(" ")
		st.write("A partir dos dados da Declaração de Nascido vivo, agora é possível predizer a Mortalidade Infantil!")

	################ SETTING DT #############
	elif choose_model == "Árvore de Decisão":
		cv_score, score, report, tree = decisionTree(X_train, X_test, y_train, y_test)
		pred = tree.predict(user_prediction_data)
		st.text("A classe predita é?")
		prob = tree.predict_proba(user_prediction_data_float)
		st.text("A probablilidade de mortalidade infantil (desfecho = 1) é: ", prob*100, "%")

	################# SETTING LR ####################
	elif choose_model == "Regressão Logística":
		cv_score, score, report, lr = decisionTree(X_train, X_test, y_train, y_test)
		pred = lr.predict(user_prediction_data)
		st.text("A classe predita é:", pred)
		prob = lr.predict_proba(user_prediction_data_float)
		st.text("A probablilidade de mortalidade infantil (desfecho = 1) é: ", prob*100, "%")
	
	################ SETTING RF ####################

	elif choose_model == "Random Forest":
		cv_score, score, report, rf = randomForest(X_train, X_test, y_train, y_test)
		pred = rf.predict(user_prediction_data)
		st.text("A classe predita é:", pred)
		prob = rf.predict_proba(user_prediction_data_float)
		st.text("A probablilidade de mortalidade infantil (desfecho = 1) é: ", prob*100, "%")

	################# SETTING XGB ####################

	elif choose_model == "XGBoost":
		cv_score, score, report, xg = xgBoost(X_train, X_test, y_train, y_test)
		pred = xg.predict(user_prediction_data)
		st.text("A classe predita é:", pred)
		prob = xg.predict_proba(user_prediction_data_float)
		st.text("A probablilidade de mortalidade infantil (desfecho = 1) é: ", prob*100, "%")
	
	################# SETTING NNT ####################
	elif choose_model == "Redes Neurais":
		cv_score, score, report, clf = neuralNet(X_train, X_test, y_train, y_test)
		pred = clf.predict(user_prediction_data)
		st.text("A classe predita é:", pred)
		prob = clf.predict_proba(user_prediction_data_float)
		st.text("A probablilidade de mortalidade infantil (desfecho = 1) é: ", prob*100, "%")


# Explicação dos Modelos
if choose_analysis == 'Explicação dos modelos':
		################ NONE #############
	if choose_model == "Nenhum":
		st.write(" ")
		st.write("A partir dos modelos preditos, agora conseguimos descobrir as variáveis que mais o explicam.")
		st.write(" ")
		st.write("Escolha seu modelo!!")

	################ SETTING DT #############
	elif choose_model == "Árvore de Decisão":
		cv_score, score, report, tree = decisionTree(X_train, X_test, y_train, y_test)
		sfs = SFS(tree, 
		   k_features=n_features, 
		   forward=True, 
		   floating=False, 
		   verbose=2,
		   scoring='accuracy',
		   cv=0)
		feature_names=X.columns
		sf = sfs.fit(X_sm, y_sm, custom_feature_names=feature_names)
		sf.subsets_

	################# SETTING LR ####################
	elif choose_model == "Regressão Logística":
		cv_score, score, report, lr = logisticRegression(X_train, X_test, y_train, y_test)
		sfs = SFS(lr, 
		   k_features=n_features, 
		   forward=True, 
		   floating=False, 
		   verbose=2,
		   scoring='accuracy',
		   cv=0)					
		feature_names=X.columns
		sf = sfs.fit(X_sm, y_sm, custom_feature_names=feature_names)
		sf.subsets_
		
	################# SETTING RF ####################

	elif choose_model == "Random Forest":
		cv_score, score, report, rf = randomForest(X_train, X_test, y_train, y_test)
		sfs = SFS(rf, 
		   k_features=n_features, 
		   forward=True, 
		   floating=False, 
		   verbose=2,
		   scoring='accuracy',
		   cv=0)
		feature_names=X.columns
		sf = sfs.fit(X_sm, y_sm, custom_feature_names=feature_names)
		sf.subsets_

	################# SETTING XGBoost ####################

	elif choose_model == "XGBoost":
		cv_score, score, report, xg = xgBoost(X_train, X_test, y_train, y_test)
		sfs = SFS(xg, 
		   k_features=n_features, 
		   forward=True, 
		   floating=False, 
		   verbose=2,
		   scoring='accuracy',
		   cv=0)
		feature_names=X.columns
		sf = sfs.fit(X_sm, y_sm, custom_feature_names=feature_names)
		sf.subsets_
	################# SETTING NNT ####################

	elif choose_model == "Redes Neurais":
		cv_score, score, report, clf = neuralNet(X_train, X_test, y_train, y_test)
		sfs = SFS(xg, 
		   k_features=n_features, 
		   forward=True, 
		   floating=False, 
		   verbose=2,
		   scoring='accuracy',
		   cv=0)
		feature_names=X.columns
		sf = sfs.fit(X_sm, y_sm, custom_feature_names=feature_names)
		sf.subsets_