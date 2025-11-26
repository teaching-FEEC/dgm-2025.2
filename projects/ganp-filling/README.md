# `Modelos generativos para Preenchimento de Lacunas em Séries Temporais de Qualidade do Ar`
# `Generative Model for Gap Filling in Air Quality Time Series Data`

## Presentation

This project originated in the context of the graduate course *IA376N - Generative AI: from models to multimodal applications*, 
offered in the second semester of 2025, at Unicamp, under the supervision of Prof. Dr. Paula Dornhofer Paro Costa, from the Department of Computer and Automation Engineering (DCA) of the School of Electrical and Computer Engineering (FEEC).

|Name  | RA | Specialization|
|:---:|:---:|:---:|
| Gabriel Caminha de Araujo Costa   | 266324  | Electrical Engineering|
| Luís Fernando Silva Lima  | 298966  | Electrical Engineering|

## Project Summary Description
Air quality monitoring is essential for obtaining data used in public health and environmental management research, especially in more arid regions with high industrial concentration. However, the data collected often has defects due to calibration and accuracy problems in the measurement sensors, poor maintenance, and communication protocol failures. In Campinas, São Paulo, fine particulate matter (PM2.5) data, provided by the São Paulo State Environmental Company (CETESB), suffer from this discontinuity, which hinders the creation of predictive models and the adoption of early strategies to mitigate problems caused by the accumulation of particulate matter, such as increased losses due to dirt in photovoltaic (PV) systems. The main motivation for this project is the application of deep learning techniques and the use of generative networks for gap filling in environmental time series datasets, with a view to their use and application for mathematical models that estimate losses due to soiling in PV systems.

The main objective of the project is to develop and evaluate the performance of a generative model adapted for time series analysis with environmental data inputs. This model should be capable of generating realistic synthetic data to fill gaps in PM2.5 data, together with contextual meteorological variables (temperature, humidity, and wind speed).

The desired output of the generative model will be a complete multivariate time series sequence, with previously missing values replaced by synthetic values that are numerically and temporally consistent with the actual input data.

## Proposed Methodology
Two data monitoring platforms will be analyzed: CETESB, which evaluates data on various pollutants for different cities in the state of São Paulo, and the Rio de Janeiro State Environmental Institute (INEA-RJ). CETESB data will be used primarily because it contains information on particulate matter in regions of relevant photovoltaic interest, while INEA data will be useful for validating the natural behavior of the data, as it has fewer measurement errors and gaps. Initially, datasets from one of the monitoring stations in Campinas for the years 2019 and 2020 will be used. The data include hourly time series of particulate matter (PM2.5), humidity, temperature, and wind speed. If more context is needed for the data, new datasets with new meteorological parameters, such as rainfall, will be studied.

Initial tests are being conducted on data using generative adversarial networks (GANs) specific to time series, such as TemporalGAN, an architecture for analyzing satellite time series data. The final architecture will be inspired by the GANFilling article, adapting its concepts from 2D (images) to 1D (time series). The Generator has a 1D U-Net architecture with convolutional and recurrent layers (LSTM/GRU), and the Discriminator is a temporal PatchGAN to evaluate the local realism of the generated data sequence. Architectures based on variational autoencoder (VAE-LSTM) models and statistical models will also be used to compare which technique excels in the final data generation. Articles will be studied to provide a basis for understanding the nature of environmental data, types of architectures suitable for time series, and models of soiling losses in photovoltaic systems.

This project will be developed in Python, using TensorFlow with the Keras API for model construction and training, Pandas and NumPy for data manipulation, Scikit-learn for preprocessing (scaling), and Matplotlib/Seaborn for visualizations, all within a Google Colab (for access to GPUs and to facilitate collaboration) or local VS Code/Jupyter environment.

## Geração dos dados sintéticos
Como formar de gerar os dados sintéticos foram escolhidos 3 modelos: XGBoost (Extreme Gradient Boosting), LSTM (Long Short-Term Memory) e TNN (Transformer Neural Network).

#### XGBoost (Extreme Gradient Boosting)

#### LSTM (Long Short-Term Memory)

#### TNN (Transformer Neural Network)
O trabalho de Orozco López, Kaplan e Linhoss (2024) se mostrou uma boa referência a ser seguida no ramo das LLM's para a predição de dados climáticos. O estudo explora o potencial das TNNs para realizar a previsão de séries temporais em múltiplas variáveis ambientais usando observações passadas e previsões meteorológicas. A arquitetura Transformer utilizada no estudo baseia-se na estrutura padrão encoder-decoder, adaptada de forma eficiente para a tarefa de rrevisão de séries temporais. O modelo emprega a arquitetura Informer no seu decoder, um diferencial metodológico fundamental, pois permite realizar previsões simultâneas de múltiplos passos futuros.

Nesse contexto, essa estrutura de Transformer foi adotada para prever as concentrações de material particulado PM2.5 e PM10 nas cidades de Dracena-SP e Brasília-DF. Como entrada do modelo foram utilizados dados de temperatura, umidade, direção do vento, velocidade do vento e dados atrasados 
em 2 semanas das variáveis a serem previstas, portanto, o contexto do Transformer era de 14 dias, além desse valor também foram realizados testes com 1, 3 e 7 dias. Além disso, durante a etapa de pré-processamento, 28 amostras no conjunto de dados foram identificadas como estando incompletas, apresentando valores ausentes. Para garantir que o conjunto de dados de entrada estivesse completo, um requisito da arquitetura Transformer, o método de imputação MissForest foi utilizado.

O conjunto de dados, uma vez imputado, foi dividido para as etapas de modelagem. Para o treinamento e validação do modelo, foram reservados 90% do total de dados. Desses, foi aplicada a divisão de 80% para o treinamento propriamente dito e 20% para a validação, sendo esta crucial para o ajuste de hiperparâmetros e controle de overfitting. O restante, que correspondeu a 10% dos dados, foi utilizado como o conjunto de teste, abrangendo um intervalo de 6 meses da série temporal, garantindo uma avaliação de desempenho imparcial em dados não observados durante a calibração.

Para a predição de PM2.5 e PM10 em Dracena-SP foram utilizados os seguintes hiperparâmetros:
- Batch size: 16;
- Contexto: 14 dias;
- Dimensão do embedding: 256;
- Quantidade de cabeças de atenção: 2;
- Número de camadas do encoder: 2;
- Número de camadas do decoder: 2;
- Taxa de dropout: 0.1;
- Learning rate: 0.0001;
- Total de 1.721.345 parâmetros.

Além disso, o RMSE (Root Mean Square Error) foi utilizado como loss function, abordando primeiramente a previsão dos valores médios de PM2.5, seu comportamento nos dados de treinamento e validação pode ser observado abaixo.

<img width="515" height="337" alt="image" src="https://github.com/user-attachments/assets/f0c20ec3-01f6-4f5d-b287-a2ba7debb8fa" />

Ambas as curvas, a Perda de Treinamento (Train Loss) e a Perda de Validação (Val Loss), exibem uma clara tendência decrescente nas aproximadamente 32 épocas. Esse comportamento indica uma boa capacidade de generalização do modelo e, crucialmente, a ausência de um overfitting significativo. Se houvesse overfitting, era esperado que a perda de validação começasse a subir dramaticamente enquanto a perda de treinamento continuaria a cair. Com o término do treinamento foram então submetidos ao modelo os dados de testes, os quais são mostrados a seguir.

<img width="890" height="378" alt="image" src="https://github.com/user-attachments/assets/a1ea8c1b-34de-4631-9e11-295fbe92dd6b" />

O modelo Transformer demonstra uma boa capacidade em seguir a tendência geral da série temporal. Durante a maior parte do período, a curva de previsão acompanha de perto a curva real, especialmente nos períodos de baixa concentração e nas variações semanais/mensais. Já para a previsão dos valores médios diários de PM10 a loss function durante o treinamento se comportou conforme a imagem abaixo.

<img width="536" height="350" alt="image" src="https://github.com/user-attachments/assets/3349f38b-3f5e-46e6-8662-cd55684fdc88" />

O padrão observado é de um aprendizado bem-sucedido e estável. As curvas de perda caem rapidamente nas épocas iniciais e, a partir da época 10, continuam a diminuir gradualmente, convergindo para a faixa de 0.5 a 0.55 MSE. É importante notar que a loss de validação se mantém em valores baixos e, em muitos momentos, abaixo da loss de treinamento. Essa relação não indica sinais de overfitting, demonstrando novamente que o modelo possui uma boa capacidade de generalização. Assim, a previsão dos valores foi gerada e é ilustrada a seguir.

<img width="891" height="378" alt="image" src="https://github.com/user-attachments/assets/56514844-f9ea-43b7-ad54-7bb955e67b32" />

O modelo Transformer demonstra uma excelente capacidade de rastrear a tendência geral e as flutuações diárias do PM10 na maior parte do período. A previsão (linha vermelha) acompanha de perto os dados reais, especialmente nas concentrações médias e baixas. A próxima etapa foi então gerar os dados de PM2.5 e PM10 para a cidade de Brasília-DF, as mesmas entradas do caso de Dracena-SP foram utilizadas, porém, foi necessário um aumento dos parâmetros do Transformer, os hiperparâmetros utilizados foram:
- Batch size: 16;
- Contexto: 14 dias;
- Dimensão do embedding: 512;
- Quantidade de cabeças de atenção: 2;
- Número de camadas do encoder: 3;
- Número de camadas do decoder: 3;
- Taxa de dropout: 0.1;
- Learning rate: 0.0001;
- Total de 9.877.377 parâmetros.

Apesar dessa necessidade de mais parâmetros a rede ainda conseguiu prever satisfatoriamente os valores de PM2.5, abaixo está ilustrado o comportamento da loss no treinamento e os valores previstos para os dados de teste.

<img width="520" height="326" alt="image" src="https://github.com/user-attachments/assets/804f1b9c-ac7b-4a41-bb83-1d03c1c4a288" />

Novamente, o treinamento do modelo Transformer foi altamente eficaz, demonstrando boa convergência e uma capacidade superior de generalização, sem mostrar sinais de overfitting, assim, permitindo gerar as previsões para os dados de testes, as quais são mostrada abaixo.

<img width="906" height="385" alt="image" src="https://github.com/user-attachments/assets/50d83a37-f28e-4669-8187-ccebe2442dbe" />

O modelo Transformer demonstra uma excelente capacidade de rastrear a tendência geral e as flutuações diárias da concentração de PM2.5. A linha de previsão (vermelha) acompanha de perto os dados reais, especialmente nas concentrações médias e baixas, mostrando um bom ajuste na maior parte do período. Utilizando os mesmos hiperparâmetros também foram previstos os dados médios de PM10, a loss de treinamento e validação é ilustrada a seguir.

<img width="514" height="333" alt="image" src="https://github.com/user-attachments/assets/d0e4a44c-3a32-43c5-9608-8d832df9e1ea" />

<img width="908" height="386" alt="image" src="https://github.com/user-attachments/assets/c56ec59e-05ac-49d5-a862-402f83e0fbee" />

É notória a convergência bem-sucedida, com as perdas de treinamento e validação diminuindo e se estabilizando em torno de 0.5 a 0.55 MSE. A proximidade das curvas e o fato de a perda de validação não subir drasticamente indicam uma boa capacidade de generalização e a ausência de overfitting significativo. Enquanto, na predição o modelo demonstra excelente acompanhamento da tendência geral e das flutuações de baixa e média intensidade. No entanto, ele subestima consistentemente a magnitude dos picos extremos, embora consiga identificar a ocorrência desses eventos.







## Link to the presentation slides
Link: https://docs.google.com/presentation/d/1kFf_0EhQC6g1P-hkWeMTLLw7tgoicSCRWhuINDGripE/edit?usp=sharing

## Bibliographic References
World Health Organization – WHO (2021). Global air quality guidelines: particulate matter (PM2.5 and PM10), ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide. Geneva: World Health Organization.

INSTITUTO DE ENERGIA E MEIO AMBIENTE - IEMA (2024). Sizing the Basic Network for Air Quality Monitoring in Brazil – Initial Scenarios.

Goodfellow, I., et al. (2014). Generative adversarial nets. Advances in neural information processing systems, 27.

Desai, A., Freeman, C., Beaver, I. and Wang, Z. (2021) TimeVAE: a variational auto-encoder for multivariate time series Generation. ICLR 2022. Available at: https://arxiv.org/abs/2111.08095.

Gonzalez-Calabuig, M., Fernández-Torres, M.Á., & Camps-Valls, G. (2025). Generative networks for spatio-temporal gap filling of Sentinel-2 reflectances. ISPRS Journal of Photogrammetry and Remote Sensing, 220, 637-648.

Toth, T. L., Kaldellis, J. K., and Mavrakis, A. (2020). ”Predicting Photovoltaic Soiling from Air Quality Measurements.” Renewable Energy, 145, 2627–2638.

Orozco López, Enrique; Kaplan, David; Linhoss, Anna. Interpretable transformer neural network prediction of diverse environmental time series using weather forecasts. Water Resources Research, v. 60, n. 10, p. e2023WR036337, 2024.
