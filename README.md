<img src='https://github.com/gcochlar/COVID19_Predict_ICU/raw/main/images/Fachada_HSL.jpg'>

# **COVID-19: ANÁLISE PREDITIVA DAS INTERNAÇÕES EM UTI NO HOSPITAL SÍRIO-LIBANÊS**
---

Bem vindo ao projeto de conclusão do ***Bootcamp de Data Science Aplicada*** da [Alura](https://www.alura.com.br).

O presente projeto utiliza as informações disponibilizadas no Kaggle pelo Hospital Sírio-Libanês para um desafio preditivo, que busca prever quais pacientes com COVID-19 necessitarão de internação em Unidades de Terapia Intensiva, recurso limitado, escasso e de alto custo.
Para isso irei utilizar das ferramentas de *Data Science* e *Machine Learning* aprendidas ao longo desse Bootcamp.

O notebook que está no presente repositório mostra todas as etapas que foram necessárias para chegar nos resultados abaixo, desde a recuperação e tratamento dos dados até os testes de diversos modelos que levaram ao modelo final.

**Recomendo sua leitura para um melhor entendimento!**

## **ESTRUTURA DO PROJETO**
---

1. [CONTEXTO](#contexto)
2. [AMBIENTE e DADOS](#dados)
3. [LIMPEZA e PREPARAÇÃO DOS DADOS](#limpeza)
4. [ANÁLISE EXPLORATÓRIA](#explora)
5. [TESTES DE MODELOS](#testes)
6. [APLICANDO O MELHOR MODELO](#melhor)
7. [CONCLUSÃO e PRÓXIMOS PASSOS](#conclusao)
8. [REFERÊNCIAS e DOCUMENTAÇÃO](#docs)

<a name='contexto'></a>
## **1. CONTEXTO**
---

A epidemia de **COVID-19** tomou o mundo inteiro de surpresa, com especial efeito no sistema de saúde como um todo.

A súbita escalada na demanda por atendimento médico exigiu uma resposta rápida por parte dos gestores, uma vez que as medidas de contenção de circulação das pessoas levariam algum tempo para surtir efeito.

Nessa situação, os esforços para o \"achatamento da curva\" tinham como principal objetivo fazer com que o número de casos não ultrapassasse a estrutura existente e permitir que essa mesma estrutura fosse aumentada onde fosse mais necessária.

<img src='https://github.com/gcochlar/Bootcamp_DataScience/raw/main/TCC/images/coronavirus-abre-en.gif' allign='right'>

Dentro dos diversos setores impactados, um dos mais críticos foi o das UTIs (Unidades de Terapia Intensiva). Além do aumento de casos, muitos pacientes infectados apresentaram necessidade de cuidados intensivos por mais tempo do que os hospitais estão habituados a oferecer.

Além disso, diversos hospitais não tinham locais adequados para garantir as condições exigidas para a instalação de novos leitos especializados e, aqueles que tinham espaço disponível, tiveram dificuldades para encontrar os profissionais habilitados ou para obter os suprimentos necessários.

Diante desse cenário, a necessidade de melhor prever a possibilidade de internações em UTIs para permitir o planejamento de ocupação e redirecionamento de pacientes cresceu exponencialmente.

O **Hospital Sírio-Libanês**, através de sua equipe de *Data Science*, trabalhou para montar um modelo de predição de necessidade de leitos de UTI com base nos dados clínicos dos pacientes admitidos com confirmação de **COVID-19**.

Esse time recebeu duas tarefas a serem cumpridas e esse desafio foi compartilhado no ***Kaggle***, plataforma que reúne a comunidade de *data scientists* e *machine learners* para troca de experiências.

**TAREFA 1:** Prever a internação em UTI de casos confirmados de **COVID-19**.

Baseado nos dados disponíveis, é factível prever quais pacientes irão necessitar de internação na unidade de tratamento intensivo? O objetivo é disponibilizar ao hospital e parceiros respostas o mais acuradas possíveis para que os recursos de UTI sejam providenciados ou a transferência do paciente seja agendada.

>>**TASK 1:** *Predict admission to the ICU of confirmed COVID-19 cases.*
>>
>>*Based on the data available, is it feasible to predict which patients will need intensive care unit support? The aim is to provide tertiary and quarternary hospitals with the most accurate answer, so ICU resources can be arranged or patient transfer can be scheduled.*

**TAREFA 2:** Prever a **NÃO** internação em UTI de casos confirmados de **COVID-19**.
Baseado nos dados disponíveis, é factível prever quais pacientes irão necessitar de internação na unidade de tratamento intensivo? O objetivo é disponibilizar para hospitais locais e temporários uma resposta suficiente, para que médicos da linha de frente possam dispensar esses pacientes e fazer acompanhamento remoto de sua evolução.

>>**TASK 2:** *Predict **NOT** admission to the ICU of confirmed COVID-19 cases.*
>>
>>*Based on the subsample of widely available data, is it feasible to predict which patients will need intensive care unit support? The aim is to provide local and temporary hospitals a good enough answer, so frontline physicians can safely discharge and remotely follow up with these patients.*

---
---
>>## **É em cima desse desafio proposto e suas tarefas que o presente estudo foi baseado.**
---
---

<a name='dados'></a>
## **2. AMBIENTE e DADOS**
---

Os dados a serem utilizados foram disponibilizados na plataforma [Kaggle](https://www.kaggle.com), em página específica para o desafio ([aqui](https://www.kaggle.com/S%C3%ADrio-Libanes/covid19/)).

Para garantir a reprodutibilidade do estudo, caso os dados sejam alterados, os mesmos foram colocados no presente repositório do **GitHub**, de onde serão importados para serem trabalhados e analisados.

<a name='limpeza'></a>
## **3. LIMPEZA e PREPARAÇÃO DOS DADOS**
---

A documentação a respeito da estrutura do *dataset* pode ser encontrada [aqui](https://www.kaggle.com/S%C3%ADrio-Libanes/covid19/).

A estrutura encontrada pode ser assim descrita:
* Identificação do paciente e janela de medição dos sinais (2 variáveis)
* Informações Demográficas (3 variáveis)
* Grupos de Doenças Prévias (9 variáveis)
* Resultados de Exames de Sangue (36 variáveis)
* Sinais Vitais (6 variáveis)
* Indicação de Internação em UTI (variável resultado)

As 42 variáveis contínuas (exames de sangue e sinais vitais) já foram normalizadas para apresentar resultados entre -1 e 1 e contam com colunas com valores estatísticos (*mean*, *median*, *max*, *min*, *diff* e *relative diff*), pois podemos ter mais de uma medição dentro de uma mesma janela horária.

A janela de medição dos sinais agrupa o conjunto de medições feitas para um paciente, o que contribui para a anonimização dos dados. Estão distribuídas da seguinte forma:
* '0-2' : Até 2 horas após a admissão do paciente
* '2-4' : Entre 2 e 4 horas da admissão do paciente
* '4-6' : Entre 4 e 6 horas da admissão do paciente
* '6-12': Entre 6 e 12 horas da admissão do paciente
* 'Above-12' : Acima de 12 horas após a admissão do paciente

A documentação informa ainda que os dados nas janelas de medição em que tivermos **`'ICU' = 1`**, ou seja, o paciente está na UTI, deverão ser desprezados, pois a medição pode ter ocorrido depois da transferência, perdendo seu efeito preditivo.

<img src='https://github.com/gcochlar/COVID19_Predict_ICU/raw/main/images/windows_0.png'>

<img src='https://github.com/gcochlar/COVID19_Predict_ICU/raw/main/images/windows_1.png'>

<a name='explora'></a>
## **4. ANÁLISE EXPLORATÓRIA**
---

A análise exploratória é mais necessária quando estamos trabalhando com dados completamente desconhecidos.

No presente estudo já tivemos uma primeira fase de tratamento dos dados feita pela própria equipe do **Hospital Sírio-Libanês**. Além disso o *dataset* foi objeto de estudo e análises exploratórias durante os módulos 4 e 5 do nosso Bootcamp. Os notebooks utilizados ao longo desses módulos podem ser encontrados [aqui](https://github.com/gcochlar/Bootcamp_DataScience/blob/main/Modulo_04/Bootcamp_Mod4_Aula05.ipynb) e [aqui](https://github.com/gcochlar/Bootcamp_DataScience/blob/main/Modulo_05/Bootcamp_Mod5_Aula06.ipynb), com todas as anotações feitas ao longo das aulas.

No presente capítulo vamos verificar apenas algumas estatísticas básicas dos dados demográficos da base fornecida, ainda em seu estado bruto.

<img src='https://github.com/gcochlar/COVID19_Predict_ICU/raw/main/images/FxEtaria.PNG'>

Em relação à distribuição por faixa etária, podemos notar um certo equilíbrio na distribuição dos pacientes totais, mas chama a atenção a representação gráfica de algo que foi amplamente divulgado em campanhas de saúde:

---
>### ***O COVID-19 ataca de forma mais severa os pacientes idosos!***
---

Isso pode ser claramente observado pelo aumento das barras azuis quanto mais nos deslocamos para a direita do gráfico (maior faixa etária), sinalizando maior necessidade de internações em UTI.

Em compensação, as barras laranja, que indicam pacientes que não precisaram de internação em UTI, ficam maiores à esquerda do gráfico, onde temos os pacientes mais jovens.

<a name='testes'></a>
## **5. TESTES DE MODELOS**
---

Vamos avaliar 4 modelos e compará-los ao nosso *baseline*, que é o modelo *dummy*. Serão:
* **`DummyClassifier`**: nosso modelo *dummy* que serve de *baseline* para comparações
* **`LogisticRegression`**: modelo de Regressão Logística
* **`DecisionTreeClassifier`**: modelo de Árvore de Decisão
* **`ExtraTreesClassifier`**: modelo que gera um conjunto randômico de Árvores de Decisão
* **`RandomForestClassifier`**: modelo que gera um conjunto randômico de Árvores de Decisão

Os modelos **`ExtraTreesClassifier`** e **`RandomForestClassifier`** são semelhantes, mas divergem na maneira com que o fator aleatório é utilizado.

No **`RandomForestClassifier`** a aleatoriedade gera árvores a partir de subconjuntos das variáveis e depois, para cada árvore, busca encontrar a variável e o valor que irá permitir uma divisão mais equilibrada dentro dos dados restantes.

Já no modelo **`ExtraTreesClassifier`**, a aleatoriedade é aplicada na geração das árvores e depois novamente para selecionar os valores de separação das variáveis que irão gerar os \galhos\. O modelo escolhe, entre os gerados, aquele que apresentar um melhor equilíbrio.

Os modelos randômicos, em função da aleatoriedade presente, reduzem significativamente o risco de **overfit** na parametrização do modelo.

<a name='melhor'></a>
## **6. APLICANDO O MELHOR MODELO**
---

Nesse capítulo pegamos o melhor modelo com os melhores parâmetros e fazemos diversas análises de performance em termos de acurácia, para checar a consistência do modelo.

Além disso, verificamos também quais foram as variáveis mais importantes para a tomada de decisão do modelo.

<a name='conclusao'></a>
## **7. CONCLUSÃO e PRÓXIMOS PASSOS**
---

Voltando aos objetivos iniciais do estudo, vamos recuperar as duas tarefas propostas pelo desafio do HSL no Kaggle.

>>**TAREFA 1:** Prever a internação em UTI de casos confirmados de **COVID-19**.
>
>Baseado nos dados disponíveis, é factível prever quais pacientes irão necessitar de internação na unidade de tratamento intensivo? O objetivo é disponibilizar ao hospital e parceiros respostas o mais acuradas possíveis para que os recursos de UTI sejam providenciados ou a transferência do paciente seja agendada.

>>**TAREFA 2:** Prever a **NÃO** internação em UTI de casos confirmados de **COVID-19**.
>
>Baseado nos dados disponíveis, é factível prever quais pacientes irão necessitar de internação na unidade de tratamento intensivo? O objetivo é disponibilizar para hospitais locais e temporários uma resposta suficiente, para que médicos da linha de frente possam dispensar esses pacientes e fazer acompanhamento remoto de sua evolução.

Baseado nos resultados do modelo final, acredito que tivemos um desempenho melhor na **TAREFA 1**, até porque, no meu entender, ela gera menos risco. **Explico:** a tarefa era para conseguir dimensionar a necessidade de UTIs e, de 115 casos em que seria necessária a internação, o modelo previu 94 (mais 2 falsos positivos). Teríamos a falta de 19 leitos.

Já a **TAREFA 2**, que envolvia a possibilidade de liberação de pacientes para acompanhamento remoto, apresentou um erro menor (dos 256 casos liberados, 21 iriam precisar de UTI) mas que poderia se mostrar mais comprometedor a partir do momento que esses pacientes precisassem retornar ao hospital, já num estado mais avançado da doença.

Concluído o estudo, acredito termos um modelo robusto o suficiente para avançarmos com os próximos passos.

**PRÓXIMOS PASSOS:**

Vejo que poderíamos prosseguir, a partir de agora, em duas frentes paralelas:

1. Recuperar uma maior quantidade de dados do período que passou desde o lançamento do desafio e verificar se o modelo mantém a consistência ou se irá precisar de maiores ajustes.

2. Revisar o modelo em relação a essas variáveis que foram desconsideradas na hora de fazer as classificações, pois a simplificação do modelo deve melhorar seu desempenho, pelo menos em termos de processamento.

<a name='docs'></a>
## **8. REFERÊNCIAS e DOCUMENTAÇÃO**
---

* [Kaggle](https://www.kaggle.com)
* [Flowing Data](https://flowingdata.com/2020/03/09/flatten-the-coronavirus-curve/?fbclid=IwAR3sG7Mkre45ZOQMH-xwWhKZzgRF6PJfydjezgPR8mS8BJ-DuwNqBHTjdUM)
* [Notebooks do Bootcamp](https://github.com/gcochlar/Bootcamp_DataScience)
* Sem esquecer todo o material disponibilizado pela Alura ao longo do Bootcamp

**BIBLIOTECAS UTILIZADAS**

* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [NumPy](https://numpy.org/)
* [SciPy](https://scipy.org/)
* [SciKit-Learn](https://scikit-learn.org/stable/)
* [Joblib](https://joblib.readthedocs.io/en/latest/)
