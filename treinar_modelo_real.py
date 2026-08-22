import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

df = pd.read_csv('data/credit_risk_dataset.csv')

colunas_selecionadas = [
    'person_age',
    'person_income',
    'loan_amnt',
    'loan_percent_income',
    'cb_person_default_on_file',
    'loan_status'
]
df_filtrado = df[colunas_selecionadas].copy()

# Removendo valores nulos
df_filtrado = df_filtrado.dropna()

# 'Y' (Yes) vira 1 (Histórico de inadimplência/Nome sujo)
# 'N' (No) vira 0 (Nome limpo no histórico)
df_filtrado['cb_person_default_on_file'] = df_filtrado['cb_person_default_on_file'].map({'Y':1, 'N': 0})
print(df_filtrado.head())

modelo = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

x = df_filtrado[['person_age', 'person_income', 'loan_amnt', 'loan_percent_income', 'cb_person_default_on_file']]
y = df_filtrado['loan_status']

x_treino, x_test, y_treino, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

modelo.fit(x_treino, y_treino)

previsoes = modelo.predict(x_test)
accuracy = accuracy_score(y_test, previsoes)
matriz = confusion_matrix(y_test, previsoes)
print(f'Accuracy do modelo: {accuracy:.4f}%')
print(f'Matriz confusion modelo: {matriz}')

joblib.dump(modelo, 'models/modelo_sistema_financeiro.joblib')
print("\nNovo modelo salvo com sucesso na pasta models/!")
