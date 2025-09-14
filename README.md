# Análise de Espectroscopia com Machine Learning

## Descrição do Projeto

Este projeto tem como objetivo analisar dados de espectroscopia de uma variedade de manga utilizando técnicas de machine learning para identificar padrões e prever atributos de qualidade como Firmness (N)', 'Dry Mass (%)', 'TSS (Brix)', 'TA (g/mL)', 'AA (mg/100g)','Weight (g)','Width (mm)','Length (mm)'. O pipeline é organizado com base no código de Andressa (mais informações abaixo), o qual está dividido em pastas específicas para cada etapa do processo incluindo, pré-processamento dos dados, redução de dimensionalidade com PCA, seleção de amostras com o método Kennard-Stone, método Jackknife para seleção de componentes e aplicação de modelos como PCR, PLSR, Random Forest, SVR e MLP.

## Funcionalidades

- **Pré-processamento**: Normalização e padronização dos dados espectroscópicos.
- **Redução de Dimensionalidade**: Aplicação de PCA para reduzir o número de features.
- **Seleção de Amostras**: Uso do método Kennard-Stone para dividir os dados em conjuntos de calibração e validação.
- **Modelos de Machine Learning**:
  - Principal Component Regression (PCR)
  - Partial Least Squares Regression (PLSR)
  - Random Forest Regression (RFR)
  - Support Vector Regression (SVR)
  - Multi-Layer Perceptron Regression (MLPR)

## Como Usar

1. **Clone o repositório**:
   ```bash
   git clone https://github.com/M00NCX/analise-sarahgit
   ```
2. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```
3. **Abra o arquivo Jupyter de cada pasta e execute para visualizar os resultados.**

## Estrutura do projeto

```
/ML-spectroscopy-analysis
│
├── /Data                                       # Dados brutos
├── /Multilayer_Perceptron_Regression           # Regressão com Multilayer Perceptron (MLPR)
├── /Principal_Components_Analysis              # Análise de Componentes Principais (PCA)
├── /Partial_Least_Squares_Regression           # Regressão por Mínimos Quadrados Parciais (PLSR)
├── /Pre-processing                             # Pré-processamento dos dados
├── /Principal_Components_Regression            # Regressão por Componentes Principais (PCR)
├── /Processed                                  # Dados processados
├── /Random_Forest_Regressor                    # Regressão por Florestas Aleatórias (RFR)
├── /Support_Vector_Machine_Regression          # Regressão por Máquinas de Vetores de Suporte (SVMR)
├── .gitignore                                  # Arquivo para ignorar arquivos desnecessários
├── README.md                                   # Documentação do projeto
└── requirements.txt                            # Dependências do projeto
```

## Licença

Este projeto está licenciado sob a [MIT License](https://choosealicense.com/licenses/mit/).

## Contato

Se tiver dúvidas ou sugestões, entre em contato:

Nome: [Adryelle](https://www.linkedin.com/in/adryelle-linhares/)

Email: [hemmolol1996@gmail.com]

GitHub: M00NCX

## Agradecimentos pelo projeto:

Nome: [Andressa](https://www.linkedin.com/in/andressa-carvalho-6b09b2312/)

Email: [acarvalho0710@gmail.com]

GitHub: xndrxssx
