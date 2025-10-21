# `Análise de Sinais de Áudio com Modelos de Mundo`
# `Audio Signal Analysis with World Models`

## Apresentação

|Nome  | RA | Especialização|
|--|--|--|
| Davi Pincinato  | 157810  | Eng. Computação |
| Henrique Parede de Souza  | 260497  | Eng. Computação|
| Isadora Minuzzi Vieira  | 290184  | Eng. Biomédica|
| Raphael Carvalho da Silva e Silva  | 205125  | Eng. Computação |



## Descrição do projeto
O projeto propõe uma nova abordagem para a análise de sinais de áudio, adaptando o conceito de "modelos de mundo" do aprendizado por reforço para um domínio de sinais. O objetivo é desenvolver um modelo de aprendizado profundo que possa aprender representações e dinâmicas de dados de áudio a partir de seus espectrogramas.

A inspiração central vem das arquiteturas DreamerV2 e Dreaming, que resulta no modelo Dreaming V2. Enquanto o DreamerV2 se destaca ao aprender comportamentos em ambientes de jogos complexos, como o Atari, usando representações discretas de estados latentes, o Dreaming V2 estende essa ideia combinando a representação discreta com um objetivo de aprendizado livre de reconstrução. A principal adaptação é substituir as imagens de jogos por espectrogramas de áudio e tratar as janelas temporais do espectrograma como "tokens" discretos. Essa "tokenização" permite que o modelo capture a dinâmica temporal do áudio de uma forma que seja compatível com arquiteturas baseadas em atenção.

## Base de referência
Este projeto é baseado em duas referências principais:

#### 1. Artigo Acadêmico: ["Dreaming V2: Reinforcement Learning with Discrete World Models without Reconstruction"](https://arxiv.org/pdf/2203.00494).

**Principais contribuições/inspirações**: O artigo apresenta o Dreaming V2, uma extensão colaborativa do DreamerV2 e Dreaming. Ele adota a representação discreta do DreamerV2 e um objetivo livre de reconstrução do Dreaming. O modelo de mundo é treinado usando um aprendizado contrastivo, que elimina a necessidade de reconstruir observações visuais complexas. Os autores demonstraram que esta abordagem alcança resultados de última geração em tarefas de braços robóticos em 3D.

#### 2. Bases de Código para Implementação:
[dreamer-torch](https://github.com/jsikyoon/dreamer-torch): Implementação em PyTorch se assemelha ao código original do DreamerV2, que foi escrito em TensorFlow. É uma referência valiosa para entender a estrutura e a lógica do modelo em um framework amplamente utilizado na comunidade de pesquisa. Os resultados demonstraram desempenho similar ao do modelo original em tarefas de controle de jogos.

[pydreamer](https://github.com/jurgisp/pydreamer): Outra reimplementação do DreamerV2 em PyTorch, que introduz algumas diferenças sutis e melhorias, como o uso de processos separados para o trainer e os workers do ambiente, permitindo que o GPU seja utilizado totalmente. Esta base de código serve como um ponto de partida para explorar abordagens ligeiramente diferentes e otimizações.

### Metodologia da adaptação e implementação
A metodologia para adaptar o Dreaming V2 para dados de áudio envolverá as seguintes etapas:

#### 1. Pré-processamento de Áudio:

- Converter os arquivos de áudio brutos em espectrogramas (por exemplo, utilizando a Transformada de Fourier de Curto Tempo - STFT).

- Normalizar e pré-processar os espectrogramas para que sejam compatíveis com a entrada do modelo.

#### 2. "Tokenização" do Espectrograma:

- Ao invés de tratar o espectrograma como uma imagem 2D, ele será dividido em janelas temporais fixas. Cada janela se tornará um "token" que representa um segmento do áudio.

- Cada "token" será codificado em um vetor de alta dimensão (embedding) que será a entrada para o modelo de mundo.

#### 3. Arquitetura do Modelo de Mundo:

- A arquitetura será baseada no Recurrent State Space Model (RSSM), que define estados latentes estocásticos e determinísticos.

- O modelo de mundo será adaptado para processar a sequência de "tokens" de espectrograma. Ele consistirá em um codificador (encoder), um modelo de dinâmica latente (RSSM) e um decodificador (decoder). No entanto, o treinamento focará no objetivo de aprendizado contrastivo, eliminando a necessidade de reconstrução completa do espectrograma.

- O codificador transformará as janelas do espectrograma em um estado latente discreto, composto por variáveis categóricas.

- O RSSM será responsável por prever o próximo estado e recompensa (neste caso, a próxima janela de espectrograma) a partir do estado atual e de uma "ação" (pode ser um placeholder ou uma variável para representar a transição temporal).

#### 4. Treinamento do Modelo:

- O modelo será treinado usando um objetivo de aprendizado contrastivo, que compara pares positivos (previsão e observação correspondente) e negativos (previsão e observações diferentes) para aprender uma representação robusta sem a necessidade de uma tarefa de reconstrução explícita.

- O aprendizado será totalmente self-supervised, focado em aprender a dinâmica do mundo de áudio.

## Cronograma

Legenda: ▓ = duração da tarefa, ⭐ = entrega

| Fase de Trabalho       | Atividades Principais                           | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|------------------------|-------------------------------------------------|---|---|---|---|---|---|---|---|---|----|----|
| Preparação & Setup     | Setup do ambiente + revisão código              | ▓ |   |   |   |   |   |   |   |   |    |    |
| Pré-processamento      | Conversão áudio → espectrograma + normalização  |   | ▓ |   |   |   |   |   |   |   |    |    |
| Pré-processamento      | Tokenização (janelas → embeddings)              |   | ▓ | ▓ |   |   |   |   |   |   |    |    |
| Modelo de Mundo        | Encoder + RSSM (ajuste usando DreamerV2)    |   |   | ▓ | ▓ |   |   |   |   |   |    |    |
| Modelo de Mundo        | Integração do Decoder / avaliação básica        |   |   |   |   | ▓ |   |   |   |   |    |    |
| **Entrega Parcial**    | Status do projeto                               |   |   |   |   |   | ⭐ |   |   |   |    |    |
| Treinamento            | Execução com DreamerV2 + ajustes leves      |   |   |   |   |   |   | ▓ | ▓ |   |    |    |
| Avaliação & Ajustes    | Análise de métricas e resultados                |   |   |   |   |   |   |   | ▓ | ▓ |    |    |
| Documentação           | Relatório + notebooks + apresentação            |   |   |   |   |   |   |   |   | ▓ | ▓  |    |
| **Entrega Final**      | Refinamento, validação final e entrega          |   |   |   |   |   |   |   |   |   |    | ⭐  |




## Tecnologias e bibliotecas utilizadas
**Linguagem**: Python

**Frameworks de Deep Learning**: PyTorch (conforme as bases de código de referência)

**Processamento de Áudio**: Librosa, Torchaudio

**Manipulação de Dados**: NumPy, Pandas

**Visualização**: Matplotlib, TensorBoard

## Links para a Apresentação

- Link para o [vídeo da apresentação](https://drive.google.com/file/d/1IFhNwxeS_8Gce3WTqXLOq8UJDLKJB7QQ/view?usp=sharing)
- Link para os [slides da apresentação](https://www.canva.com/design/DAGzF_vtvEE/6c1_5Sw-mUuLSqV6HMjP9Q/edit?utm_content=DAGzF_vtvEE&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)



Abstract

Summary of the objective, methodology and results obtained (in submission E2 it is possible to report partial results). Suggested maximum of 100 words.
Problem Description / Motivation

Description of the generating context of the project theme. Motivation for addressing this project theme.
Objective

Description of what the project aims to do.
It is possible to specify a general objective and specific objectives of the project.
Methodology

Clearly and objectively describe, citing references, the methodology proposed to achieve the project objectives.
Describe datasets used.
Cite reference algorithms.
Justify the reasons for the chosen methods.
Point out relevant tools.
Describe the evaluation methodology (how will it be assessed whether the objectives were met or not?).
Datasets and Evolution

List the datasets used in the project.
For each dataset, include a mini-table in the model below and then provide details on how it was analyzed/used, as in the example below.
Dataset	Web Address	Descriptive Summary
Dataset Title	http://base1.org/	Brief summary (two or three lines) about the dataset.
Provide a description of what you concluded about this dataset. Suggested guiding questions or information to include:

What is the dataset format, size, type of annotation?
What transformations and preprocessing were done? Cleaning, re-annotation, etc.
Include a summary with descriptive statistics of the dataset(s).
Use tables and/or charts to describe the main aspects of the dataset that are relevant to the project.
Workflow

Use a tool that allows you to design the workflow and save it as an image (e.g., Draw.io). Insert the image in this section.
You may choose to use a workflow manager (Sacred, Pachyderm, etc.), in which case use the manager to generate a diagram for you.
Remember that the goal of drawing the workflow is to help anyone who wishes to reproduce your experiments.
Experiments, Results, and Discussion of Results

In the partial project submission (E2), this section may contain partial results, explorations of implemented solutions, and
discussions about such experiments, including decisions to change the project trajectory or the description of new experiments as a result of these explorations.
In the final project submission (E3), this section should list the main results obtained (not necessarily all), which best represent the fulfillment of the project objectives.
The discussion of results may be carried out in a separate section or integrated into the results section. This is a matter of style.
It is considered fundamental that the presentation of results should not serve as a treatise whose only purpose is to show that "a lot of work was done."
What is expected from this section is that it presents and discusses only the most relevant results, highlighting the strengths and/or limitations of the methodology, emphasizing aspects of performance, and containing content that can be classified as organized, didactic, and reproducible sharing of knowledge relevant to the community.
Conclusion

The Conclusion section should recover the main information already presented in the report and point to future work.
In the partial project submission (E2), it may contain information about which steps or how the project will be conducted until its completion.
In the final project submission (E3), the conclusion is expected to outline, among other aspects, possibilities for the project’s continuation.
Bibliographic References

Indicate in this section the bibliographic references adopted in the project.


## Referências:
HA, David; SCHMIDHUBER, Jürgen. World Models. arXiv:1803.10122, 2018.
https://arxiv.org/abs/1803.10122

HAFNER, Danijar et al. DreamerV2: Mastering Atari with Discrete World Models. arXiv:2010.02193, 2020.
https://arxiv.org/abs/2010.02193

HAFNER, Danijar et al. Learning Latent Dynamics for Planning from Pixels (PlaNet). ICML, 2019.
https://arxiv.org/abs/1811.04551

OORD, Aaron van den et al. Neural Discrete Representation Learning (VQ-VAE). NeurIPS, 2017.
https://arxiv.org/abs/1711.00937

PRABHU, Kundan Kumar et al. Autoregressive Spectrogram Inpainting with Time–Frequency Transformers. arXiv preprint, 2021.
https://arxiv.org/abs/2104.03976

WANG, Yuxuan et al. Tacotron: Towards End-to-End Speech Synthesis. Interspeech, 2017.
https://arxiv.org/abs/1703.10135

PANAYOTOV, Vassil et al. LibriSpeech: An ASR Corpus. ICASSP, 2015.
https://www.openslr.org/12

Mozilla. Common Voice Dataset.
https://commonvoice.mozilla.org
