# `Modelos generativos aplicados à previsão do acúmulo de sujeira em sistemas fotovoltaicos`
# `Generative models applied to predicting soiling in photovoltaic systems`

## Presentation

This project originated in the context of the graduate course *IA376N - Generative AI: from models to multimodal applications*, 
offered in the second semester of 2025, at Unicamp, under the supervision of Prof. Dr. Paula Dornhofer Paro Costa, from the Department of Computer and Automation Engineering (DCA) of the School of Electrical and Computer Engineering (FEEC).

|Name  | RA | Specialization|
|:---:|:---:|:---:|
| Gabriel Caminha de Araujo Costa   | 266324  | Electrical Engineering|
| Luís Fernando Silva Lima  | 298966  | Electrical Engineering|

## Project Summary Description
Soiling is a common type of loss studied in photovoltaic (PV) systems and results from the accumulation of dirt on the surface of the modules. This problem affects the performance of PV systems, especially in arid regions with high concentrations of dust and particles, increasing maintenance costs for cleaning and reducing the Performance Ratio (PR) of photovoltaic plants. It is therefore necessary to understand the physical behavior of deposition over time to mitigate losses in system generation. In the literature, there are two mathematical models that describe dirt losses in photovoltaic systems in a reproducible way. The Kimber model (2006), a pioneer in the field, uses precipitation time series and empirical deposition coefficients to indicate daily measured dirt (SR). The HSU model (2018), on the other hand, incorporates atmospheric data on fine and coarse particles (PM2.5 and PM10) as a starting point for defining the rate of dirt deposition on surfaces. Some more recent models, such as the one presented by Toth (2020), have sought to add more variables, such as wind speed, to provide a broader context for the data.

Therefore, it is correct to say that air quality monitoring is essential for obtaining data used to operate photovoltaic systems, especially in more arid regions with high industrial concentration. However, the data collected often has flaws due to calibration and accuracy problems with measurement sensors, inadequate maintenance, and communication protocol failures. In Campinas, São Paulo, for example, fine particle (PM2.5) data provided by the São Paulo State Environmental Company (CETESB) suffers from this discontinuity, which hinders the creation of predictive models and the adoption of strategies to mitigate losses caused by dirt. In the figure below, you can see a time series from the database with several points of failure, highlighting the difficulty of this type of study.

<img width="891" height="298" alt="image" src="https://github.com/user-attachments/assets/0707db24-f53e-4526-8b9a-10025f6c983f" />


The main motivation for this project is to verify the feasibility of using deep learning techniques and generative networks to predict data intervals in environmental time series, with a view to their use and application in mathematical models that estimate losses due to soiling in photovoltaic systems. This model should be capable of generating realistic synthetic data to fill gaps in PM2.5 and PM10 data, together with contextual meteorological variables (temperature, humidity, wind speed, and wind direction). The desired output of the generative model will be a complete multivariate time series sequence, with previously missing values replaced by synthetic values that are numerically and temporally consistent with the actual input data.

## Proposed Methodology
Six years of hourly particulate matter data were collected, covering the years 2018 to 2023. Due to initially poor training results with national databases, we turned to the API of a German startup called PVRADAR, which provides reconstructed historical data in long, uninterrupted series from other pollutants measured by NASA's MERRA-2 satellite. From this new database, the training and test data were divided.

The modeling process is focused on predicting the PM2.5 and PM10 particulate matter concentrations in the cities of Dracena-SP and Brasília-DF. The total dataset was divided into three main subsets for model development and testing: Training and Validation, and Test. For the Training and Validation phases, 90% of the total data was reserved, with 80% of that subset used for training the model and 20% for validation. The remaining 10% of the data was set aside as the test set, corresponding to a 6-month interval of the time series, used for impartial final model evaluation. The model aims to generate forecasts for the mean daily concentrations of PM2.5 and PM10 in both Dracena-SP and Brasília-DF. 

The predicted PM2.5 and PM10 data will subsequently be utilized as crucial input for a HSU model to forecast the soiling ratio in photovoltaic (PV) systems. This integration is vital because the concentration of airborne particulate matter, which is strongly correlated with the soiling ratio, directly impacts the efficiency and power output of solar panels. By feeding the accurate, time-series predictions from the models into the HSU system, the overall forecasting framework can provide more reliable estimates of power degradation due to accumulated dust, enabling better operational planning and maintenance scheduling for PV installations.

## Workflow

<img width="1157" height="581" alt="Diagrama sem nome drawio (5)" src="https://github.com/user-attachments/assets/d2fa6200-e221-4a88-8e44-2c2a98e9fc66" />

## Synthetic Data Generation
Three models were chosen to generate the synthetic data: XGBoost (Extreme Gradient Boosting), LSTM (Long Short-Term Memory), and TNN (Transformer Neural Network).

#### XGBoost (Extreme Gradient Boosting)

#### LSTM (Long Short-Term Memory)

#### TNN (Transformer Neural Network)
The work of Orozco López, Kaplan, and Linhoss (2024) proved to be a good reference to follow in the field of LLMs for the prediction of climatic data. The study explores the potential of TNNs to perform time series forecasting across multiple environmental variables using past observations and weather forecasts. The Transformer architecture used in the study is based on the standard encoder-decoder structure, efficiently adapted for the task of time series forecasting. The model employs the Informer architecture in its decoder, a fundamental methodological difference, as it allows for simultaneous forecasting of multiple future steps.

In this context, this Transformer structure was adopted to predict particulate matter concentrations of PM2.5 and PM10 in the cities of Dracena-SP and Brasília-DF. The model inputs included data on temperature, humidity, wind direction, wind speed, and lagged data of the variables to be predicted by 2 weeks; therefore, the Transformer's context was 14 days. In addition to this value, tests were also performed with 1, 3, and 7 days. Furthermore, during the preprocessing step, 28 samples in the dataset were identified as incomplete, presenting missing values. To ensure the input dataset was complete, a requirement of the Transformer architecture, the MissForest imputation method was used.

The dataset, once imputed, was divided for the modeling steps. For model training and validation, 90% of the total data was reserved. Of this, the division of 80% was applied for training itself and 20% for validation, which was crucial for hyperparameter tuning and control of overfitting. The remainder, which corresponded to 10% of the data, was used as the test set, covering a 6-month interval of the time series, ensuring an impartial performance evaluation on data not observed during calibration.

For the prediction of PM2.5 and PM10 in Dracena-SP, the following hyperparameters were used:
- Batch size: 16;
- Context: 14 dias;
- Embedding dimension:: 256;
- Number of attention heads: 2;
- Number of encoder layers: 2;
- Number of decoder layers: 2;
- Dropout rate: 0.1;
- Learning rate: 0.0001;
- Total of 1.721.345 parameters.

Additionally, the RMSE (Root Mean Square Error) was used as the loss function, first addressing the prediction of the mean values of PM2.5, whose behavior in the training and validation data can be observed below.

<img width="515" height="337" alt="image" src="https://github.com/user-attachments/assets/f0c20ec3-01f6-4f5d-b287-a2ba7debb8fa" />

Both curves, the Train Loss and the Val Loss, show a clear decreasing trend over approximately 32 epochs. This behavior indicates a good capacity for model generalization and, crucially, the absence of significant overfitting. If overfitting were present, it would be expected that the validation loss would start to rise dramatically while the training loss would continue to fall. Upon completion of training, the test data, shown below, was then submitted to the model.

<img width="890" height="378" alt="image" src="https://github.com/user-attachments/assets/a1ea8c1b-34de-4631-9e11-295fbe92dd6b" />

The Transformer model demonstrates a good ability to follow the general trend of the time series. During most of the period, the prediction curve closely follows the real curve, especially in periods of low concentration and in weekly/monthly variations. For the prediction of mean daily PM10 values, the loss function during training behaved as shown in the image below.

<img width="536" height="350" alt="image" src="https://github.com/user-attachments/assets/3349f38b-3f5e-46e6-8662-cd55684fdc88" />

The observed pattern is one of successful and stable learning. The loss curves drop rapidly in the initial epochs and, from epoch 10, continue to decrease gradually, converging to the 0.5 to 0.55 MSE range. It is important to note that the validation loss remains at low values and, in many moments, below the training loss. This relationship does not indicate signs of overfitting, demonstrating again that the model possesses good generalization capacity. Thus, the prediction of the values was generated and is illustrated below.

<img width="891" height="378" alt="image" src="https://github.com/user-attachments/assets/56514844-f9ea-43b7-ad54-7bb955e67b32" />

The Transformer model demonstrates an excellent ability to track the general trend and daily fluctuations of PM10 during most of the period. The prediction (red line) closely follows the real data, especially in medium and low concentrations. The next step was then to generate the PM2.5 and $\text{PM10}$ data for the city of Brasília-DF. The same inputs as the Dracena-SP case were used, however, an increase in the Transformer's parameters was necessary. The hyperparameters used were:
- Batch size: 16;
- Context: 14 days;
- Embedding dimension: 512;
- Number of attention heads: 2;
- Number of encoder layers: 3;
- Number of decoder layers: 3;
- Dropout rate: 0.1;
- Learning rate: 0.0001;
- Total of 9.877.377 parameters.

Despite this need for more parameters, the network still managed to predict the $\text{PM2.5}$ values satisfactorily. Illustrated below is the loss behavior during training and the predicted values for the test data.

<img width="520" height="326" alt="image" src="https://github.com/user-attachments/assets/804f1b9c-ac7b-4a41-bb83-1d03c1c4a288" />

Again, the training of the Transformer model was highly effective, demonstrating good convergence and a superior generalization capacity, without showing signs of overfitting, thus allowing the predictions for the test data to be generated, which are shown below.

<img width="906" height="385" alt="image" src="https://github.com/user-attachments/assets/50d83a37-f28e-4669-8187-ccebe2442dbe" />

The Transformer model demonstrates an excellent ability to track the general trend and daily fluctuations of PM2.5 concentration. The prediction line (red) closely follows the real data, especially in medium and low concentrations, showing a good fit over most of the period. Using the same hyperparameters, the mean PM10 data was also predicted, with the training and validation loss illustrated below.

<img width="514" height="333" alt="image" src="https://github.com/user-attachments/assets/d0e4a44c-3a32-43c5-9608-8d832df9e1ea" />

<img width="908" height="386" alt="image" src="https://github.com/user-attachments/assets/c56ec59e-05ac-49d5-a862-402f83e0fbee" />

The successful convergence is notable, with training and validation losses decreasing and stabilizing around 0.5 to 0.55 MSE. The proximity of the curves and the fact that the validation loss does not rise drastically indicate a good capacity for generalization and the absence of significant overfitting. Meanwhile, in the prediction, the model demonstrates excellent tracking of the general trend and low-to-medium intensity fluctuations. However, it consistently underestimates the magnitude of the extreme peaks, although it manages to identify the occurrence of these events.

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
