#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Bibliotecas para el preprocesamiento y modelado
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Bibliotecas para modelado y evaluación
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.utils import resample


# In[2]:


data = pd.read_csv('/datasets/Churn.csv')


# In[3]:


# Mostrar las primeras filas del conjunto de datos
print(data.head())

# Verificar la estructura general del conjunto de datos
print(data.info())

# Resumen estadístico de las columnas numéricas
print(data.describe())

# Comprobar valores únicos en características categóricas
print(data['Geography'].value_counts())
print(data['Gender'].value_counts())

# Inspeccionar el balance de clases en la variable objetivo
print(data['Exited'].value_counts(normalize=True))


# In[4]:


# Eliminar columnas irrelevantes ya que estas columnas son variables para identificar al cliente y no aportan un valor como tal para identificar si el ciente se va o no
data = data.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# La columna Tunure tiene valores nulos por lo que se usara la media para imputar estos valores
data['Tenure'].fillna(data['Tenure'].median(), inplace=True)

# Hay columnas categoricas las cuales hay que cambair a variables numericas.
data = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=True)

# Escalamiento de características numéricas
scaler = StandardScaler()
numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
data[numerical_features] = scaler.fit_transform(data[numerical_features])


# In[5]:


sns.set(style='whitegrid')
plt.figure(figsize=(10, 6))

# 1. Histogramas para las características numéricas
data[numerical_features].hist(bins=20, figsize=(14, 10))
plt.suptitle('Distribución de Características Numéricas', fontsize=16)
plt.show()

# 2. Boxplots para detectar valores atípicos
plt.figure(figsize=(14, 8))
for i, feature in enumerate(numerical_features):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x=data[feature])
    plt.title(f'Boxplot de {feature}')
plt.tight_layout()
plt.show()

# 3. Matriz de correlación
plt.figure(figsize=(12, 8))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlación', fontsize=16)
plt.show()

# 4. Distribución de las características categóricas
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(x='Geography_Germany', data=data, ax=axes[0])
axes[0].set_title('Clientes en Alemania')
sns.countplot(x='Gender_Male', data=data, ax=axes[1])
axes[1].set_title('Distribución de Género (Masculino)')
plt.tight_layout()
plt.show()

# 5. Distribución de la variable objetivo
plt.figure(figsize=(6, 4))
sns.countplot(x='Exited', data=data)
plt.title('Distribución de la Variable Objetivo (Exited)')
plt.show()

# 6. Comparación de características importantes por clase
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for i, feature in enumerate(numerical_features):
    sns.histplot(data=data, x=feature, hue='Exited', multiple='stack', ax=axes[i//3, i%3])
    axes[i//3, i%3].set_title(f'{feature} por Estado de Cliente')
plt.tight_layout()
plt.show()


# In[6]:


# Separar características y objetivo
features = data.drop('Exited', axis=1)
target = data['Exited']

# Divición del conjunto de datos en entrenamiento y prueba (75% - 25%)
features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=0.25, random_state=12345)

# Entrenar un modelo de Regresión Logística (sin tratar el desequilibrio de clases)
model = LogisticRegression(random_state=12345)
model.fit(features_train, target_train)

# Predicciones en el conjunto de prueba
predicted_valid = model.predict(features_valid)

#Calculo de as métricas F1 y AUC-ROC
f1 = f1_score(target_valid, predicted_valid)
auc_roc = roc_auc_score(target_valid, model.predict_proba(features_valid)[:, 1])

# Resultados:
print(f"F1 Score: {f1:.2f}")
print(f"AUC-ROC Score: {auc_roc:.2f}")
print("\nReporte de clasificación:\n")
print(classification_report(target_valid, predicted_valid))


# In[7]:


# Enfoque 1: Ajuste de pesos de clase con 'balanced'
models = {
    'Logistic Regression': LogisticRegression(random_state=12345, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(random_state=12345, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=12345, class_weight='balanced')
}

# Función para entrenar y evaluar modelos con ajuste de peso de clases
def train_and_evaluate(models, features_train, target_train, features_valid, target_valid):
    results = {}
    for model_name, model in models.items():
        # Entrenar el modelo
        model.fit(features_train, target_train)
        # Predicciones
        predicted_valid = model.predict(features_valid)
        # Calcular métricas
        f1 = f1_score(target_valid, predicted_valid)
        auc_roc = roc_auc_score(target_valid, model.predict_proba(features_valid)[:, 1])
        results[model_name] = {'F1 Score': f1, 'AUC-ROC': auc_roc}
        print(f"{model_name} - F1 Score: {f1:.2f}, AUC-ROC: {auc_roc:.2f}")
        print(classification_report(target_valid, predicted_valid))
        print("\n")
    return results

print("Resultados con ajuste de pesos ('balanced'):")
results_balanced = train_and_evaluate(models, features_train, target_train, features_valid, target_valid)

# Enfoque 2: Submuestreo de la clase mayoritaria

# Combinación de los datos de entrenamiento
train_data = pd.concat([features_train, target_train], axis=1)

# Separación de clases mayoritaria y minoritaria
majority_class = train_data[train_data['Exited'] == 0]
minority_class = train_data[train_data['Exited'] == 1]

# Submuestrear la clase mayoritaria
majority_class_downsampled = resample(majority_class,
                                      replace=False,
                                      n_samples=len(minority_class),
                                      random_state=12345)

# Conjunto de datos balanceado
balanced_train_data = pd.concat([majority_class_downsampled, minority_class])

# Separación las características y el objetivo en el conjunto balanceado
features_train_balanced = balanced_train_data.drop('Exited', axis=1)
target_train_balanced = balanced_train_data['Exited']

print("Resultados usando Submuestreo manual:")
results_downsampled = train_and_evaluate(models, features_train_balanced, target_train_balanced, features_valid, target_valid)


# In[9]:


features = data.drop('Exited', axis=1)
target = data['Exited']

# Divicipon del conjunto de datos en entrenamiento y prueba
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.25, random_state=12345)

# Balanceo mediante submuestreo
test_data = pd.concat([features_test, target_test], axis=1)

# Separación las clases en el conjunto de prueba para el submuestreo
majority_class_test = test_data[test_data['Exited'] == 0]
minority_class_test = test_data[test_data['Exited'] == 1]

# Submuestreo la clase mayoritaria en el conjunto de prueba
majority_class_downsampled_test = resample(majority_class_test,
                                           replace=False,
                                           n_samples=len(minority_class_test),
                                           random_state=12345)

# Conjunto de prueba balanceado
balanced_test_data = pd.concat([majority_class_downsampled_test, minority_class_test])

# Separarción características y objetivo en el conjunto de prueba balanceado
features_test_balanced = balanced_test_data.drop('Exited', axis=1)
target_test_balanced = balanced_test_data['Exited']

# Entrenamiento del mejor modelo que en este caso fue Bosque Aleatorio con el conjunto de entrenamiento balanceado mediante submuestreo
best_model = RandomForestClassifier(random_state=12345)
best_model.fit(features_train_balanced, target_train_balanced)

# Predicciones en el conjunto de prueba balanceado
predicted_test = best_model.predict(features_test_balanced)

# Métricas F1 y AUC-ROC en el conjunto de prueba final
f1_final = f1_score(target_test_balanced, predicted_test)
auc_roc_final = roc_auc_score(target_test_balanced, best_model.predict_proba(features_test_balanced)[:, 1])

# Resultados Finales
print(f"F1 Score Final: {f1_final:.2f}")
print(f"AUC-ROC Score Final: {auc_roc_final:.2f}")
print("\nReporte de clasificación final:\n")
print(classification_report(target_test_balanced, predicted_test))

