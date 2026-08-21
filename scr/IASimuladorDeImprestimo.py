import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from random import randint, choice
import joblib


class IAFinanceira:
    nome_arquivo = 'models/modelo_sistema_financeiro.joblib'
    arquivo_dados = 'data/historico_credito.csv'

    @staticmethod
    def __pontuacao(idade:int, salario:float, valor:float, nome_limpo:int):
        score = 0

        if nome_limpo == 1:
            score += 50
        else:
            score -= 50


        if salario >= 10000:
            score += 20
        elif salario >= 5000:
            score += 10

        relacao = valor / salario

        if relacao <= 3:
            score += 30
        elif relacao <= 6:
            score += 20
        elif relacao <= 10:
            score += 5
        else:
            score -= 30

        if 25 <= idade <= 60:
            score += 10
        elif 18 <= idade < 25:
            score += 5
        return 1 if score >= 60 else 0

    def __init__(self):
        self.__modelo = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.__treinada = False
        self.verificar()

    def __treinar_ia(self):
        x = []
        y = []
        for c in range(1, 7000):
            idade = randint(18, 80)
            salario = randint(1200, 50000)
            valor_solicitado = randint(5000, 200000)
            nome_limpo = choice([0, 1])
            resultado = self.__pontuacao(idade, salario, valor_solicitado, nome_limpo)
            x.append([idade, salario, valor_solicitado, nome_limpo])
            y.append(resultado)
        x_treino, x_test, y_treino, y_test = train_test_split(x, y,
                                                               test_size=0.2,
                                                              random_state=42,
                                                              stratify=y)
        self.__modelo.fit(x_treino, y_treino)
        previsoes = self.__modelo.predict(x_test)
        acuarry = accuracy_score(y_test, previsoes)
        confusion = confusion_matrix(y_test, previsoes)
        print(f'Accuracy: {acuarry:.4f}')
        print(f'Confusion matrix: {confusion}')
        self.__treinada = True

    def prever(self, idade:int, salario:float, valor:float, nome_limpo:int):
        resultado = self.__modelo.predict([[idade, salario, valor, nome_limpo]])
        return resultado[0] == 1 #Retorna 1 se for aprovado.

    def __salvar_modelo(self):
        joblib.dump(self.__modelo, self.nome_arquivo)

    def verificar(self):
        if os.path.exists(self.nome_arquivo):
            self.__modelo = joblib.load(self.nome_arquivo)
            self.__treinada = True
            print('Modelo carregado com sucesso!')
        else:
            self.__treinar_ia()
            self.__salvar_modelo()
            print('Modelo treinado e salvo com sucesso!')

    def calcular_probabilidade_aprovado(self, idade:int, salario:float, valor:float, nome_limpo:int):
        resultado = self.__modelo.predict_proba([[idade, salario, valor, nome_limpo]])
        return resultado[0][1]

    def calcular_probabilidade_reprovado(self, idade:int, salario:float, valor:float, nome_limpo:int):
        resultado = self.__modelo.predict_proba([[idade, salario, valor, nome_limpo]])
        return resultado[0][0]

    def testar_modelo(self):
        self.__treinar_ia()

    def __atualizar_treinamento_da_ia(self):
        pass
