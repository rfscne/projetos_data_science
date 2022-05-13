#!/usr/bin/env python
# coding: utf-8

# In[2426]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Preparação dos dados

# In[2427]:


df = pd.read_csv(r'd:\dataset\credit\CreditScoring.csv')


# In[2428]:


df.head()


# In[2429]:


df.columns = df.columns.str.lower()
df.head()


# Status está como numérico, converter para categórico

# In[2430]:


status_values = {
    1: 'ok',
    2: 'default',
    0: 'unk'
}
df.status = df.status.map(status_values)
df.head()


# In[2431]:


home_values = {
    1: 'rent',
    2: 'owner',
    3: 'private',
    4: 'ignore',
    5: 'parentes',
    6: 'other',
    0: 'unk'
}
df.home = df.home.map(home_values)


# In[2432]:


marital_values = {
    1: 'single',
    2: 'married',
    3: 'window',
    4: 'separated',
    5: 'divorced',
    0: 'unk'
}
df.marital = df.marital.map(marital_values)


# In[2433]:


record_values = {
    1: 'no',
    2: 'yes',
    0: 'unk'
}
df.records = df.records.map(record_values)


# In[2434]:


job_values = {
    1: 'fixed',
    2: 'partime',
    3: 'freelance',
    4: 'others',
    0: 'unk'
}
df.job = df.job.map(job_values)


# In[2435]:


df.head()


# Após o processo anterior, todas as variáveis categóricas são strings. 
# Agora analisar as variáveis numéricas: 

# In[2436]:


df.describe().round()


# 99999999 significa valores faltantes (substituir por NAN)

# In[2437]:


for c in['income', 'assets', 'debt']:
    df[c] = df[c].replace(to_replace= 99999999, value=np.nan)


# In[2438]:


df.isnull().sum()


# In[2439]:


df.describe().round


# Vamos analisar a variável alvo

# In[2440]:


df.status.value_counts()


# Como tem somente uma linha de status que é desconhecida podemos eliminá-la

# In[2441]:


df =df[df.status != 'unk']


# In[2442]:


df.isnull().sum()


# Preparar os dados para treinamento. 
# Dividir o dataset (treino/validação/teste)
# Aplicar one-hot encoding para características categóricas. 

# In[2443]:


from sklearn.model_selection import train_test_split


# In[2444]:


df_train_full, df_test = train_test_split(df, test_size = 0.2, random_state=11)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state =11)


# In[2445]:


y_train = (df_train.status == 'default').values
y_val = (df_val.status == 'default').values


# In[2446]:


del df_train['status']
del df_val['status']


# In[2447]:


len(df_train), len(df_val), len(df_test)


# Para OHE, utilizar DicVectorizer

# In[2448]:


from sklearn.feature_extraction import DictVectorizer


# Substituir os valores faltantes por 0

# In[2449]:


df_train.isnull().sum()


# In[2450]:


dict_train = df_train.fillna(0).to_dict(orient='records')
dict_val = df_val.fillna(0).to_dict(orient='records')


# In[2451]:


dict_train[0]


# In[2452]:


dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(dict_train)
X_val = dv.transform(dict_val)


# Árvore de Decisão

# Classificador DecisionTreeClassifier, métrica: AUC (Area Under Curve)

# In[2453]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score


# Iniciar com os parâmetros default da árvore
# (Obtenção do modelo)

# In[2454]:


dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)


# Usar predict_proba para conseguir as predições (probabilidades)

# In[2455]:


y_pred = dt.predict_proba(X_train)[:,1]
roc_auc_score(y_train, y_pred)


# Predição no dataset de avaliação (X_val)

# In[2456]:


y_pred = dt.predict_proba(X_val)[:,1]
roc_auc_score(y_val, y_pred)


# Caso de overfitting -> dados de treino performam perfeitamente, mas falham na validação

# Mudar os parâmetros: restringir o tamanho da árvore a 2 níveis: 

# In[2457]:


dt = DecisionTreeClassifier(max_depth=2)
dt.fit(X_train, y_train)

y_pred = dt.predict_proba(X_train)[:,1]
auc = roc_auc_score(y_train, y_pred)
print('train auc: %.3f' %auc)

y_pred = dt.predict_proba(X_val)[:,1]
auc = roc_auc_score(y_val, y_pred)
print('val auc: %.3f' % auc)


# Resultado melhor que o anterior

# Tunning os parâmetros

# In[2458]:


dt = DecisionTreeClassifier(max_depth=6)
dt.fit(X_train, y_train)


# In[2459]:


y_pred = dt.predict_proba(X_val)[:,1]


# In[2460]:


roc_auc_score(y_val, y_pred)


# In[2461]:


for depth in [1, 2, 3, 4, 5, 6, 10, 15, 20, None]:
    dt = DecisionTreeClassifier(max_depth=depth)
    dt.fit(X_train, y_train)
    y_pred = dt.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, y_pred)
    print('%4s -> %.3f' % (depth, auc))


# In[2462]:


for m in [1, 5, 10, 15, 20, 50, 100, 200]:
    dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=m)
    dt.fit(X_train, y_train)
    y_pred = dt.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, y_pred)
    print('%s -> %.3f' %(m,auc))


# In[2463]:


for m in[4,5, 6]:
    print('depth: %s' %m)
    for  s in[1, 5, 10, 15, 20, 50, 100, 200]:
        dt = DecisionTreeClassifier(max_depth=m, min_samples_leaf=s)
        dt.fit(X_train, y_train)
        y_pred = dt.predict_proba(X_val)[:,1]
        auc = roc_auc_score(y_val, y_pred)
        print('%s->%.3f' %(s,auc))
    print()


# In[2464]:


for m in[1,5,10,15,20,50,100,200]:
    dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=m)
    dt.fit(X_train, y_train)
    y_pred = dt.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, y_pred)
    print('%s-> %.3f' %(m, auc))


# In[2465]:


dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict_proba(X_val)[:,1]
roc_auc_score(y_val, y_pred_dt)


# In[2466]:


from sklearn.metrics import roc_curve


# In[2467]:


fpr, tpr, _ = roc_curve(y_val, y_pred_dt)
plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, color='black')
plt.plot([0,1], [0,1], color = 'black', lw=0.7, linestyle='dashed', alpha=0.5)

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt. title ('ROC curve')
plt.show()


# Floresta Randômica

# In[2468]:


from sklearn.ensemble import RandomForestClassifier


# In[2469]:


rf = RandomForestClassifier(n_estimators = 10, random_state=3)
rf.fit(X_train, y_train)
y_pred = rf.predict_proba(X_val)[:,1]
roc_auc_score(y_val, y_pred)


# Variando o número de árvores

# In[2470]:


aucs = []
for i in range(10, 201, 10):
    rf = RandomForestClassifier(n_estimators=i, random_state=3)
    rf.fit(X_train, y_train)
    y_pred = rf.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, y_pred)
    print('%s->%.3f' %(i, auc))
    aucs.append(auc)


# In[2471]:


plt.figure(figsize=(6,4))
plt.plot(range(10, 201, 10), aucs, color='black')
plt.xticks(range(0, 201, 50))
plt.title("Number of trees vs AUC")
plt.xlabel('Number of trees')
plt.ylabel('AUC')
#plt.savefig('06_random_forest_n_estimators.svg')

plt.show()


# Testando max_depth

# In[2472]:


all_aucs = {}
for depth in [5,10,20]: #analisa profundidade (depth)
    print('depth: %s' %depth)
    aucs=[]
    
    for i in range(10, 201, 10): 
        rf = RandomForestClassifier(n_estimators=i, max_depth=depth, random_state=1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict_proba(X_val)[:,1]
        auc = roc_auc_score(y_val, y_pred)
        print('%s->%.3f' %(i, auc))
        aucs.append(auc)
    all_aucs[depth] = aucs
    print()


# In[2473]:


plt.figure(figsize=(6,4))
num_trees = list(range(10, 201, 10))
plt.plot(num_trees, all_aucs[5], label='depth=5', color='black', linestyle='dotted')
plt.plot(num_trees, all_aucs[10], label='depth=10', color='black', linestyle='dashed')
plt.plot(num_trees, all_aucs[20], label='depth=20', color='black', linestyle='solid')

plt.xticks(range(0, 201, 50))
plt.legend()

plt.title('Number of trees vs AUC')
plt.xlabel('Number of trees')
plt.ylabel('AUC')
plt.show()


# Variando min_samples_leaf

# In[2474]:


all_aucs={}
for m in[3,5,10]: #analisa min_samples_leaf
    print('min_samples_leaf:%s' %m)
    aucs=[]
    for i in range(10, 201, 20): 
        rf = RandomForestClassifier(n_estimators=i, max_depth=10, min_samples_leaf=m, random_state=1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict_proba(X_val)[:,1]
        auc = roc_auc_score(y_val, y_pred)
        print('%s->%.3f' %(i, auc))
        aucs.append(auc)
        all_aucs[m]= aucs
        print()


# In[2475]:


plt.figure(figsize=(6, 4))

num_trees = list(range(10, 201, 20))

plt.plot(num_trees, all_aucs[3], label='min_samples_leaf=3', color='black', linestyle='dotted')
plt.plot(num_trees, all_aucs[5], label='min_samples_leaf=5', color='black', linestyle='dashed')
plt.plot(num_trees, all_aucs[10], label='min_samples_leaf=10', color='black', linestyle='solid')
    
plt.xticks(range(0, 201, 50))
plt.legend()

plt.title('Number of trees vs AUC')
plt.xlabel('Number of trees')
plt.ylabel('AUC')

# plt.savefig('ch06-figures/06_random_forest_n_estimators_sample_leaf.svg')

plt.show()


# Melhor AUC encontrado com o número de árvores = 200

# Modelo final: 

# In[2476]:


rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=1)
rf.fit(X_train, y_train)


# In[2477]:


y_pred_rf = rf.predict_proba(X_val)[:,1]
roc_auc_score(y_val, y_pred_rf)


# In[2478]:


plt.figure(figsize=(5, 5))

fpr, tpr, _ = roc_curve(y_val, y_pred_rf)
plt.plot(fpr, tpr, color='black')

fpr, tpr, _ = roc_curve(y_val, y_pred_dt)
plt.plot(fpr, tpr, color='black', linestyle='dashed')

plt.plot([0, 1], [0, 1], color='black', lw=0.7, linestyle='dashed', alpha=0.5)

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('ROC curve')

plt.show()


# XGBoost

# In[2479]:


import xgboost as xgb


# In[2480]:


dtrain = xgb.DMatrix(X_train, label = y_train, feature_names = dv.feature_names_)
dval = xgb.DMatrix(X_val, label=y_val, feature_names = dv.feature_names_)


# In[2481]:


xgb_params = {
    'eta': 0.3,
    'max_depth': 6, 
    'min_child_weight': 1, 
    
    'objective': 'binary:logistic', 
    'nthread': 8, 
    'seed': 1
}


# In[2482]:


model = xgb.train(xgb_params, dtrain, num_boost_round=10)


# In[2483]:


y_pred = model.predict(dval)
y_pred[:10]


# In[2484]:


roc_auc_score(y_val, y_pred)


# In[2485]:


watchlist = [(dtrain, 'train'), (dval, 'val')]


# In[2486]:


xgb_params = {
    'eta': 0.3,
    'max_depth': 6, 
    'min_child_weight': 1, 
    
    'objective': 'binary:logistic', 
    'eval_metric': 'auc',
    'nthread': 8, 
    'seed': 1
}


# In[2487]:


model = xgb.train(xgb_params, dtrain, num_boost_round=100, evals=watchlist, verbose_eval=10)


# Para capturar a saída e colocar o resultado em output 
# %%capture , 
# parse_xgb_output

# In[2488]:


get_ipython().run_cell_magic('capture', 'output', 'model = xgb.train(xgb_params, dtrain, num_boost_round=100, evals=watchlist, verbose_eval=5)')


# In[2489]:


def parse_xgb_output(output):
    tree = []
    aucs_train = []
    aucs_val = []

    for line in output.stdout.strip().split('\n'):
        it_line, train_line, val_line = line.split('\t')

        it = int(it_line.strip('[]'))
        train = float(train_line.split(':')[1])
        val = float(val_line.split(':')[1])

        tree.append(it)
        aucs_train.append(train)
        aucs_val.append(val)

    return tree, aucs_train, aucs_val


# Para plotar os scores

# In[2490]:


tree, aucs_train, aucs_val = parse_xgb_output(output)


# In[2491]:


plt.figure(figsize=(6, 4))

plt.plot(tree, aucs_train, color='black', linestyle='dashed', label='Train AUC')
plt.plot(tree, aucs_val, color='black', linestyle='solid', label='Validation AUC')
plt.xticks(range(0, 101, 25))

plt.legend()

plt.title('XGBoost: number of trees vs AUC')
plt.xlabel('Number of trees')
plt.ylabel('AUC')

# plt.savefig('ch06-figures/06_xgb_default.svg')

plt.show()


# Tuning do parâmetro eta

# In[2492]:


get_ipython().run_cell_magic('capture', 'output', "#Para eta = 0.3\nxgb_params = {\n    'eta': 0.3,\n    'max_depth': 6,\n    'min_child_weight': 1,\n    'objective': 'binary:logistic',\n    'eval_metric': 'auc', \n    'nthread': 8, \n    'seed': 1,\n}\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=500, verbose_eval=10,\n                 evals=watchlist)")


# In[2493]:


tree, _, aucs_val_eta_03 = parse_xgb_output(output)
print(max(aucs_val_eta_03))
print(max(zip(aucs_val_eta_03, tree)))


# In[2494]:


get_ipython().run_cell_magic('capture', 'output', "#Para eta = 0.1\nxgb_params = {\n    'eta': 0.1,\n    'max_depth': 6,\n    'min_child_weight': 1,\n    'objective': 'binary:logistic',\n    'eval_metric': 'auc', \n    'nthread': 8, \n    'seed': 1,\n}\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=500, verbose_eval=10,\n                 evals=watchlist)")


# In[2495]:


tree, _, aucs_val_eta_01 = parse_xgb_output(output)
print(max(aucs_val_eta_01))
print(max(zip(aucs_val_eta_01, tree)))


# In[2496]:


get_ipython().run_cell_magic('capture', 'output', "#Para eta = 0.05,\nxgb_params = {\n    'eta': 0.05,\n    'max_depth': 6,\n    'min_child_weight': 1,\n    'objective': 'binary:logistic',\n    'eval_metric': 'auc', \n    'nthread': 8, \n    'seed': 1,\n}\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=500, verbose_eval=10,\n                 evals=watchlist)")


# In[2497]:


tree, _, aucs_val_eta_005 = parse_xgb_output(output)
print(max(aucs_val_eta_005))
print(max(zip(aucs_val_eta_005, tree)))


# In[2498]:


get_ipython().run_cell_magic('capture', 'output', "#Para eta = 0.01,\nxgb_params = {\n    'eta': 0.01,\n    'max_depth': 6,\n    'min_child_weight': 1,\n    'objective': 'binary:logistic',\n    'eval_metric': 'auc', \n    'nthread': 8, \n    'seed': 1,\n}\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=500, verbose_eval=10,\n                 evals=watchlist)")


# In[2499]:


tree, _, aucs_val_eta_001 = parse_xgb_output(output)
print(max(aucs_val_eta_001))
print(max(zip(aucs_val_eta_001, tree)))


# In[2500]:


plt.figure(figsize=(6, 4))

plt.plot(tree, aucs_val_eta_03, color='black', linestyle='solid', label='eta=0.3')
plt.plot(tree, aucs_val_eta_01, color='black', linestyle='dashed', label='eta=0.1')
# plt.plot(tree, aucs_val_eta_005, color='grey', linestyle='solid', label='eta=0.05')
# plt.plot(tree, aucs_val_eta_001, color='grey', linestyle='dashed', label='eta=0.01')

plt.xticks(range(0, 501, 100))

plt.legend()

plt.title('The effect of eta on model performance')
plt.xlabel('Number of trees')
plt.ylabel('AUC (validation)')

# plt.savefig('ch06-figures/06_xgb_eta.svg')

plt.show()


# In[2501]:


plt.figure(figsize=(6, 4))

plt.plot(tree, aucs_val_eta_01, color='grey', linestyle='dashed', label='eta=0.1')
plt.plot(tree, aucs_val_eta_005, color='black', linestyle='solid', label='eta=0.05')
plt.plot(tree, aucs_val_eta_001, color='black', linestyle='dashed', label='eta=0.01')

plt.xticks(range(0, 501, 100))

plt.legend()

plt.title('The effect of eta on model performance')
plt.xlabel('Number of trees')
plt.ylabel('AUC (validation)')

# plt.savefig('ch06-figures/06_xgb_eta_2.svg')

plt.show()


# Tuning max_depth

# #Para max_depth = 3, com eta = 0.1

# In[2502]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.1,\n    'max_depth': 3,\n    'min_child_weight': 1,\n\n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n    'nthread': 8,\n    'seed': 1,\n}\n\nmodel = xgb.train(xgb_params, dtrain,\n                  num_boost_round=500, verbose_eval=10,\n                  evals=watchlist)")


# In[2503]:


tree, _, aucs_val_depth3 = parse_xgb_output(output)
print(max(aucs_val_depth3))
print(max(zip(aucs_val_depth3, tree)))


# In[2504]:


get_ipython().run_cell_magic('capture', 'output', "# Para max_depth=10\nxgb_params = {\n    'eta': 0.1,\n    'max_depth': 10,\n    'min_child_weight': 1,\n\n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n    'nthread': 8,\n    'seed': 1,\n}\n\nmodel = xgb.train(xgb_params, dtrain,\n                  num_boost_round=500, verbose_eval=10,\n                  evals=watchlist)")


# In[2505]:


tree, _, aucs_val_depth10 = parse_xgb_output(output)
print(max(aucs_val))
print(max(zip(aucs_val_depth10, tree)))


# depth=3 é melhor do que depth=6 e depth=10. Tentar com depth=4 para ver se é melhor do que 3

# In[2506]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.1,\n    'max_depth': 4,\n    'min_child_weight': 1,\n\n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n    'nthread': 8,\n    'seed': 1,\n}\n\nmodel = xgb.train(xgb_params, dtrain,\n                  num_boost_round=500, verbose_eval=10,\n                  evals=watchlist)")


# In[2507]:


tree, _, aucs_val_depth4 = parse_xgb_output(output)
print(max(aucs_val_depth4))
print(max(zip(aucs_val_depth4, tree)))


# In[2508]:


plt.figure(figsize=(6, 4))

plt.plot(tree, aucs_val_depth3, color='black', linestyle='dashed', label='max_depth=3')
plt.plot(tree, aucs_val_depth4, color='grey', linestyle='dashed', label='max_depth=4')
plt.plot(tree, aucs_val_eta_01, color='black', linestyle='solid', label='max_depth=6')
plt.plot(tree, aucs_val_depth10, color='grey', linestyle='solid', label='max_depth=10')

plt.ylim(0.75, 0.845)
plt.xlim(-10, 510)
plt.xticks(range(0, 501, 100))

plt.legend()

plt.title('The effect of max_depth on model performance')
plt.xlabel('Number of trees')
plt.ylabel('AUC (validation)')

# plt.savefig('ch06-figures/06_xgb_depth.svg')

plt.show()


# max_depth =3 é o melhor

# Tuning min_child_weight

# In[2509]:


get_ipython().run_cell_magic('capture', 'output', "#Para o valor default min_child_weight=1\nxgb_params = {\n    'eta': 0.1,\n    'max_depth': 3,\n    'min_child_weight': 1,\n\n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n    'nthread': 8,\n    'seed': 1,\n}\n\nmodel = xgb.train(xgb_params, dtrain,\n                  num_boost_round=500, verbose_eval=10,\n                  evals=watchlist)")


# In[2510]:


tree, _, aucs_val_mcw1 = parse_xgb_output(output)
print(max(aucs_val_mcw1))
print(max(zip(aucs_val_mcw1, tree)))


# In[2511]:


get_ipython().run_cell_magic('capture', 'output', "# Para min_child_weight = 10\nxgb_params = {\n    'eta': 0.1,\n    'max_depth': 3,\n    'min_child_weight': 10,\n\n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n    'nthread': 8,\n    'seed': 1,\n}\n\nmodel = xgb.train(xgb_params, dtrain,\n                  num_boost_round=500, verbose_eval=10,\n                  evals=watchlist)")


# In[2512]:


tree, _, aucs_val_mcw10 = parse_xgb_output(output)
print(max(aucs_val_mcw10))
print(max(zip(aucs_val_mcw10, tree)))


# In[2513]:


get_ipython().run_cell_magic('capture', 'output', "# Para min_child_weight = 30\nxgb_params = {\n    'eta': 0.1,\n    'max_depth': 3,\n    'min_child_weight': 30,\n\n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n    'nthread': 8,\n    'seed': 1,\n}\n\nmodel = xgb.train(xgb_params, dtrain,\n                  num_boost_round=500, verbose_eval=10,\n                  evals=watchlist)")


# In[2514]:


tree, _, aucs_val_mcw30 = parse_xgb_output(output)
print(max(aucs_val_mcw30))
print(max(zip(aucs_val_mcw30, tree)))


# In[2515]:


plt.figure(figsize=(6, 4))

plt.plot(tree, aucs_val_mcw1, color='black', linestyle='solid', label='min_child_weight=1')
plt.plot(tree, aucs_val_mcw10, color='grey', linestyle='solid', label='min_child_weight=10')
plt.plot(tree, aucs_val_mcw30, color='black', linestyle='dashed', label='min_child_weight=30')

plt.ylim(0.82, 0.84)
plt.xlim(0, 510)
plt.xticks(range(0, 501, 100))
plt.yticks(np.linspace(0.82, 0.84, 5))

plt.legend()

plt.title('The effect of min_child_weight on model performance')
plt.xlabel('Number of trees')
plt.ylabel('AUC (validation)')

# plt.savefig('ch06-figures/06_xgb_mcw.svg')

plt.show()


# Analisando o número de árvores

# In[2516]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.1,\n    'max_depth': 3,\n    'min_child_weight': 1,\n\n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n    'nthread': 8,\n    'seed': 1,\n}\n\nmodel = xgb.train(xgb_params, dtrain,\n                  num_boost_round=500, verbose_eval=10,\n                  evals=watchlist)")


# In[2517]:


tree, _, aucs_val = parse_xgb_output(output)
print(max(aucs_val))


# In[2518]:


max(zip(aucs_val, tree))


# In[2519]:


plt.figure(figsize=(6, 4))

plt.plot(tree, aucs_val, color='black', linestyle='solid')


plt.ylim(0.80, 0.84)
plt.xlim(0, 510)
plt.xticks(range(0, 501, 100))
plt.yticks(np.linspace(0.80, 0.84, 9))


plt.vlines(180, 0, 1, color='grey', linestyle='dashed', linewidth=0.9)

plt.title('Selecting the number of trees')
plt.xlabel('Number of trees')
plt.ylabel('AUC (validation)')

# plt.savefig('ch06-figures/06_xgb_number_trees.svg')

plt.show()


# Modelo final (com parâmetros otimizados)

# In[2520]:


xgb_params = {
    'eta': 0.1,
    'max_depth': 3,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
}

model = xgb.train(xgb_params, dtrain,
                  num_boost_round=180, verbose_eval=10,
                  evals=watchlist)


# In[2521]:


#predição no dataset de validação
y_pred_xgb = model.predict(dval)


# In[2522]:


roc_auc_score(y_val, y_pred_xgb)


# In[2523]:


#Calculo das curvas roc das 3 estratégias
print(roc_auc_score(y_val, y_pred_dt))
print(roc_auc_score(y_val, y_pred_rf))
print(roc_auc_score(y_val, y_pred_xgb))


# In[2524]:


plt.figure(figsize=(5, 5))

fpr, tpr, _ = roc_curve(y_val, y_pred_xgb)
plt.plot(fpr, tpr, color='black')

fpr, tpr, _ = roc_curve(y_val, y_pred_rf)
plt.plot(fpr, tpr, color='grey', linestyle='dashed', alpha=0.9)

fpr, tpr, _ = roc_curve(y_val, y_pred_dt)
plt.plot(fpr, tpr, color='grey', linestyle='dashed', alpha=0.9)

plt.plot([0, 1], [0, 1], color='black', lw=0.7, linestyle='dashed', alpha=0.5)

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('ROC curve')

plt.show()


# Teste final utilizando o dataset de treinamento e validação juntos (y_train_full) e o 
# dataset de teste (y_test)

# In[2525]:


y_train_full = (df_train_full.status == 'default').values
y_test = (df_test.status == 'default').values

del df_train_full['status']
del df_test['status']


# In[2526]:


dict_train_full = df_train_full.fillna(0).to_dict(orient='records')
dict_test = df_test.fillna(0).to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train_full = dv.fit_transform(dict_train_full)
X_test = dv.transform(dict_test)


# In[2527]:


rf_final = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=1)
rf_final.fit(X_train_full, y_train_full)

y_pred_rf = rf.predict_proba(X_test)[:, 1]


# In[2528]:


dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full, feature_names=dv.feature_names_)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=dv.feature_names_)

xgb_params = {
    'eta': 0.1,
    'max_depth': 3,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
}

num_trees = 160

xgb_final = xgb.train(xgb_params, dtrain_full, num_boost_round=num_trees)


# In[2529]:


y_pred_xgb = xgb_final.predict(dtest)


# In[2530]:


print(roc_auc_score(y_test, y_pred_rf))
print(roc_auc_score(y_test, y_pred_xgb))


# Cálculo de feature importance

# Árvores de decisão

# In[2531]:


dt.feature_importances_


# In[2532]:


list(zip(dv.feature_names_, dt.feature_importances_))


# In[2533]:


importances = list(zip(dv.feature_names_, dt.feature_importances_))

df_importance = pd.DataFrame(importances, columns=['feature', 'gain'])
df_importance = df_importance.sort_values(by='gain', ascending=False)
df_importance


# In[2534]:


df_importance = df_importance[df_importance.gain > 0]


# In[2535]:


num = len(df_importance)
plt.barh(range(num), df_importance.gain[::-1])
plt.yticks(range(num), df_importance.feature[::-1])

plt.show()


# Floresta Randômica

# In[2536]:


rf.feature_importances_


# In[2537]:


importances = list(zip(dv.feature_names_, rf.feature_importances_))

df_importance = pd.DataFrame(importances, columns=['feature', 'gain'])
df_importance = df_importance.sort_values(by='gain', ascending=False)
df_importance


# In[2538]:



df_importance = df_importance[df_importance.gain > 0.01]


# In[2539]:


num = len(df_importance)
plt.barh(range(num), df_importance.gain[::-1])
plt.yticks(range(num), df_importance.feature[::-1])

plt.show()


# XGBoost

# In[2540]:


scores = model.get_score(importance_type='gain')
scores = sorted(scores.items(), key=lambda x: x[1])
list(reversed(scores))


# In[2541]:


scores = model.get_score(importance_type='weight')
scores = sorted(scores.items(), key=lambda x: x[1])
list(reversed(scores))


# In[2542]:


names = [n for (n, s) in scores]
scores = [s for (n, s) in scores]


# In[2543]:


plt.figure(figsize=(6, 8))

plt.barh(np.arange(len(scores)), scores)
plt.yticks(np.arange(len(names)), names)

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




