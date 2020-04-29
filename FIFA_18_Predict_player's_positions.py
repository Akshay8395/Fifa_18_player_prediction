import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import accuracy_score 
from sklearn import metrics


df = pd.read_csv('CompleteDataset.csv')

#consider columns which are required for predicting the data
col = ['Aggression','Crossing', 'Curve', 'Dribbling', 'Finishing',
       'Free kick accuracy', 'Heading accuracy', 'Long shots','Penalties', 'Shot power', 'Volleys', 
       'Short passing', 'Long passing',
       'Interceptions', 'Marking', 'Sliding tackle', 'Standing tackle',
       'Strength', 'Vision', 'Acceleration', 'Agility', 
       'Reactions', 'Stamina', 'Balance', 'Ball control','Composure','Jumping', 
       'Sprint speed', 'Positioning','Preferred Positions']
df = df[col]
df['Preferred Positions'] = df['Preferred Positions'].str.strip()
df = df[df['Preferred Positions'] != 'GK']
positions = df['Preferred Positions'].str.split().apply(lambda x: x[0]).unique()
# Creating a new data frmae
df_new = df.copy()
df_new.drop(df_new.index, inplace=True)

for i in positions:
    df_temp = df[df['Preferred Positions'].str.contains(i)]
    df_temp['Preferred Positions'] = i
    df_new = df_new.append(df_temp, ignore_index=True)
cols = [col for col in df_new.columns if col not in ['Preferred Positions']]

for i in cols:
    df_new[i] = df_new[i].apply(lambda x: eval(x) if isinstance(x,str) else x)
figure, ax = plt.subplots()
data_frame = df_new[df_new['Preferred Positions'] == 'RW'].iloc[::200,:-1]
data_frame.T.plot.line(color = 'black', figsize = (15,9), legend = False, ylim = (0, 110), title = "RW's attributes distribution", ax=ax)

ax.set_xlabel('Attributes')
ax.set_ylabel('Rating')

ax.set_xticks(np.arange(len(cols)))
ax.set_xticklabels(labels = cols, rotation=90)

for l in ax.lines:
    l.set_linewidth(1)

ax.axvline(0, color='red', linestyle='-')   
ax.axvline(14.9, color='red', linestyle='-')

ax.axvline(15, color='blue', linestyle='-')
ax.axvline(28, color='blue', linestyle='-')

ax.text(7.5, 100, 'Attack Attributes', color = 'red', weight = 'bold')
ax.text(22.5, 100, 'Defend Attributes', color = 'blue', weight = 'bold')

#Grpahy after Normalization
data_frame = df_new[df_new['Preferred Positions'] == 'RW'].iloc[::200,:-1]
Normalized_Dataframe = data_frame.div(data_frame.sum(axis=1),axis=0)

figure, ax = plt.subplots()
Normalized_Dataframe.T.plot.line(color = 'black', figsize = (15,9), legend = False, ylim = (0,0.08), title = "RW's attributes distribution", ax=ax)


ax.set_xlabel('Attributes')
ax.set_ylabel('Rating')

ax.set_xticks(np.arange(len(cols)))
ax.set_xticklabels(labels = cols, rotation=90)

for l in ax.lines:
    l.set_linewidth(1)

ax.axvline(0, color='red', linestyle='-')   
ax.axvline(14.9, color='red', linestyle='-')

ax.axvline(15, color='blue', linestyle='-')
ax.axvline(28, color='blue', linestyle='-')

ax.text(7.5, 0.07, 'Attack Attributes', color = 'red', weight = 'bold')
ax.text(22.5, 0.07, 'Defend Attributes', color = 'blue', weight = 'bold')

#Predicting binary tragets with logistic regression:
Normalized_dataframe = df_new.iloc[:,:-1].div(df_new.iloc[:,:-1].sum(axis=1), axis=0)
# put 1 for attack positions and 0 for defense positions
mapping = {'ST': 1, 'RW': 1, 'LW': 1, 'RM': 1, 'CM': 1, 'LM': 1, 'CAM': 1, 'CF': 1, 
           'CDM': 0, 'CB': 0, 'LB': 0, 'RB': 0, 'RWB': 0, 'LWB': 0}
Normalized_dataframe['Preferred Positions'] = df_new['Preferred Positions']
Normalized_dataframe = Normalized_dataframe.replace({'Preferred Positions': mapping})
#split train and test dataset:
X_train, X_test, y_train, y_test = train_test_split(Normalized_dataframe.iloc[:,:-1], 
                                                    Normalized_dataframe.iloc[:,-1] ,random_state=0)
#apply logistic regrssion on traninig set
model_train = LogisticRegression().fit(X_train, y_train)
y_pred = model_train.predict(X_test) 
accuracy = accuracy_score(y_pred,y_test)
print ('Logistic Regression Accuracy: {}'.format(accuracy))

#finding coefficient for each attribue so for better acuracy i can select top 10 features
Coef_list = list(sorted(zip(X_train.columns, abs(model_train.coef_[0])),key=lambda x: -x[1]))
Coef_table = pd.DataFrame(np.array(Coef_list).reshape(-1,2), columns = ['Attributes', 'Coef'])

print (Coef_table)

# cosider top 10 feature and train your model to get accuracy
top_attributes = Coef_table[:10]['Attributes'].tolist() 

Model_train = LogisticRegression().fit(X_train[top_attributes], y_train)
accuracy = Model_train.score(X_test[top_attributes], y_test)
print ('Logistic Regression Accuracy (10 features): {}'.format(accuracy))

#PCA impact on model accuracy
#HeatMap
f, ax = plt.subplots(figsize=(22, 22))

plt.title('Pearson Correlation of Player attributes')

sns.heatmap(df_new.corr(),linewidths=0.25,vmax=1.0,annot=True)

cov_mat = np.cov(df_new.iloc[:,:-1].T)
values, vectors = np.linalg.eig(cov_mat)

# Calculation of Explained Variance from the eigenvalues
tot = sum(values)
var_exp = [(i/tot)*100 for i in sorted(values, reverse=True)] # Individual explained variance
cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance

print(list(zip(range(29),cum_var_exp)))

plt.figure(figsize=(10, 10))
plt.bar(range(len(var_exp)), var_exp, alpha=0.3333, label='individual explained variance')
plt.step(range(len(var_exp)), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()

pca = PCA(n_components=17)

X_train, X_test, y_train, y_test = train_test_split(df_new.iloc[:,:-1], df_new.iloc[:,-1], random_state=0)

X_train = pca.fit_transform(X_train)
X_train = pd.DataFrame(X_train)

x_test = pca.transform(X_test)

Model_train = LogisticRegression().fit(X_train, y_train)
Accuracy= Model_train.score(x_test, y_test)
print ('Logistic Regression Accuracy with PCA (17 components):',Accuracy)

#Predict the best position for the player with the logistic Regression:
data_frame_all = df_new.copy()
#Mapping all the  14 postions:
mapping = {'ST': 0, 'RW': 1, 'LW': 2, 'RM': 3, 'CM': 4, 'LM': 5, 'CAM': 6, 
               'CF': 7, 'CDM': 8, 'CB': 9, 'LB': 10, 'RB': 11, 'RWB': 12, 'LWB': 13}
data_frame_all = data_frame_all.replace({'Preferred Positions': mapping})

#split train and test dataset:
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(data_frame_all.iloc[:,:-1],
                                                                    data_frame_all.iloc[:,-1],test_size=0.50 ,random_state=0)

#apply logistic regrssion on traninig set
model_train = LogisticRegression().fit(X_train_all, y_train_all)
Accuracy = model_train.score(X_test_all, y_test_all)
print ('Logistic Regression Accuracy: {}'.format(Accuracy))

#Predict the best position for the player with the Neuaral Networks:
model_train = MLPClassifier(random_state=0).fit(X_train_all, y_train_all)
Accuracy = model_train.score(X_test_all, y_test_all)
print ('Neural Networks Accuracy: {}'.format(Accuracy))


#Predict the best position for the player with the Random Forest:
model_train = RandomForestClassifier(random_state=0).fit(X_train_all, y_train_all)
Accuracy = model_train.score(X_test_all, y_test_all)
print ('Random Forest  Accuracy: {}'.format(Accuracy))
