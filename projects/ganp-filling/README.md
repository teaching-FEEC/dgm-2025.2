# `Redes Adversárias Generativas para Preenchimento de Lacunas em Séries Temporais de Qualidade do Ar`
# `Generative Adversarial Networks for Gap Filling in Air Quality Time Series Data`

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

## Schedule
| Stage | Duration	| Description
|:---:|:---:|:---:|
| Literature review and data processing | 1-2 weeks (starting on September 17) | Theoretical deep dive into GANs for time series, consolidation, and final processing of datasets for feature creation.|
| Training of models (c-GANs and VAE-LSTM) | 3-4 weeks (starting on September 30) | Construction and training of generative models (generator and discriminator), testing with data samples, verification of metrics and output data quality. |
| Analysis and validation of final results, conclusions, and suggestions for improvement | 2-3 weeks (starting October 22) | Execution of evaluation tests (qualitative and quantitative) and comparison between trained models. |
| Writing of the final article/report | 1-2 weeks (starting November 14) | Organization of data in the final document on GitHub. |

## Link to the presentation slides
Link: https://docs.google.com/presentation/d/1n6z4C6x5140AZevYEBDL8OJBNCzGOKROG6ExoKdsg9Y/edit?usp=sharing

## Bibliographic References
World Health Organization – WHO (2021). Global air quality guidelines: particulate matter (PM2.5 and PM10), ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide. Geneva: World Health Organization.

INSTITUTO DE ENERGIA E MEIO AMBIENTE - IEMA (2024). Sizing the Basic Network for Air Quality Monitoring in Brazil – Initial Scenarios.

Goodfellow, I., et al. (2014). Generative adversarial nets. Advances in neural information processing systems, 27.

Desai, A., Freeman, C., Beaver, I. and Wang, Z. (2021) TimeVAE: a variational auto-encoder for multivariate time series Generation. ICLR 2022. Available at: https://arxiv.org/abs/2111.08095.

Gonzalez-Calabuig, M., Fernández-Torres, M.Á., & Camps-Valls, G. (2025). Generative networks for spatio-temporal gap filling of Sentinel-2 reflectances. ISPRS Journal of Photogrammetry and Remote Sensing, 220, 637-648.

Toth, T. L., Kaldellis, J. K., and Mavrakis, A. (2020). ”Predicting Photovoltaic Soiling from Air Quality Measurements.” Renewable Energy, 145, 2627–2638.
