# `Análise de Sinais de Áudio com Modelos de Mundo`
# `Audio Signal Analysis with World Models`

## Apresentação

Este projeto teve origem no contexto do curso de pós-graduação IA376N - IA Generativa: dos modelos às aplicações multimodais, oferecido no segundo semestre de 2025, na Unicamp, sob a orientação da Prof.ª Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia da Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

|Nome  | RA | Especialização|
|--|--|--|
| Davi Pincinato  | 157810  | Eng. Computação |
| Henrique Parede de Souza  | 260497  | Eng. Computação|
| Isadora Minuzzi Vieira  | 290184  | Eng. Biomédica|
| Raphael Carvalho da Silva e Silva  | 205125  | Eng. Computação |

## Resumo (parcial)

Este projeto propõe adaptar modelos de mundo (World Models) à tarefa de síntese de áudio por meio de “sonhos”, tendo como base a arquitetura DreamerV2, originalmente desenvolvida para aprendizado de políticas em ambientes como o Atari. A ideia central é que a mesma estrutura que permite ao Dreamer aprender e planejar em espaços latentes pode ser explorada para modelar a dinâmica temporal de sinais sonoros. A partir de espectrogramas de áudios, treinamos um VQ-VAE para discretizar os espectrogramas, em seguida, um modelo de mundo com RSSM com estados determinísticos e estocásticos discretos para capturar suas dependências temporais. Na Etapa 2 (E2), concluímos o pré-processamento, o treinamento do VQ-VAE e iniciamos o treinamento conjunto de encoder, RSSM e decoder com perdas de reconstrução e divergência KL. Resultados parciais mostram reconstruções consistentes e uso estável do codebook, evidenciando o potencial do paradigma de World Models para geração de áudio coerente a partir de representações latentes aprendidas, análogo ao modo como o Dreamer aprende dinâmicas visuais em jogos do Atari.

## Descrição do projeto
Modelos de mundo (World Models) surgiram da área de aprendizado por reforço (RL) como uma forma de aprender representações latentes das dinâmicas do ambiente [HA et al. (2018)]. Ao invés de reagir apenas a observações imediatas, um modelo de mundo aprende a prever e “imaginar” futuros estados no seu próprio espaço latente, permitindo planejar e aprender políticas de forma mais eficiente.

Entre esses modelos, as arquiteturas Dreamer e DreamerV2 [HAFNER et al. (2020)] se destacam ao combinar um modelo de mundo latente com aprendizado de políticas inteiramente nesse espaço, dispensando a reconstrução pixel a pixel das observações visuais. Enquanto o Dreamer se destaca ao aprender comportamentos em ambientes de jogos complexos, como o Atari, usando representações discretas de estados latentes, o DreamerV2 estende essa ideia combinando a representação discreta com um objetivo de aprendizado livre de reconstrução.

Neste projeto, propomos transportar o conceito de modelos de mundo para o domínio do áudio, buscando explorar sua capacidade de previsão, completude de sequências (por exemplo, “ba-ta” → “ta”), síntese condicionada, robustez em reconhecimento automático de fala (ASR) e aprendizado self-supervised. Assim como o DreamerV2 aprende a prever e planejar em espaços latentes no domínio visual, nossa proposta visa desenvolver um modelo capaz de aprender representações latentes e dinâmicas temporais de sinais de áudio.

A principal adaptação consiste em substituir as imagens de jogos pelos espectrogramas de áudio, e em discretizar esses espectrogramas em tokens, tornando as sequências resultantes compatíveis com arquiteturas temporais baseadas em atenção e objetivos contrastivos. Essa tokenização elimina a necessidade de reconstrução direta do sinal espectral, reduzindo custos computacionais e favorecendo a escalabilidade e eficiência do treinamento. Dessa forma, o modelo pode aprender a capturar as estruturas e transições subjacentes dos sons, abrindo caminho para representações mais ricas e generalizáveis no aprendizado de áudio.

## Objetivos
O objetivo geral deste projeto consiste no treino um modelo de mundo para espectrogramas de áudio capaz de prever/imaginar a evolução de espectrogramas tokenizados. Para o cumprimento deste objetivo, são instanciados os seguintes objetivos específicos:

- Definição de dataset e pré-processamento, incluindo separação de amostras de áudio não validadas e transformação para espectrogramas.
- Proposta e implementação de um modelo VQ-VAE [OORD et al. (2017)] para tokenização dos espectrogramas de áudio.
- Treinamento do modelo de mundo (encoder + RSSM com estados discretos + decoder) para reconstrução/likelihood sobre tokens.
- Definir métricas para qualidade de tokenização, uso do codebook e compreender a capacidade preditiva do world model.
- Investigar objetivo contrastivo (sem reconstrução explícita) e integração do ator-crítico ao estado latente (DreamerV2).


## Metodologia
A metodologia para adaptar o Dreaming V2 para dados de áudio envolverá as seguintes etapas:

### 1. Pré-processamento de Áudio:

- Converter os arquivos de áudio brutos em espectrogramas utilizando a Transformada de Fourier de Curto Tempo (STFT): 
    - Esta técnica decompõe o sinal temporal em representações tempo-frequência, permitindo capturar tanto a evolução temporal quanto o conteúdo espectral do áudio.
    - A STFT é especialmente adequada para este projeto pois mantém a localização temporal das características acústicas, facilitando o aprendizado de dependências sequenciais pelo RSSM.
    - Implementado utilizando `torchaudio` e `librosa`.

- Normalizar e pré-processar os espectrogramas para que sejam compatíveis com a entrada do modelo VQ-VAE: 
    - **Conversão para escala log-mel**: Transformação para escala mel (que aproxima a percepção auditiva humana) e aplicação de logaritmo para comprimir a faixa dinâmica, tornando características sutis mais visíveis ao modelo.
    - **Normalização de amplitude**: Aplicação de normalização z-score para garantir que os valores do espectrograma estejam em uma faixa adequada, evitando problemas de convergência durante o treinamento.

- Avaliação preliminar dos espectrogramas gerados:
    **Visualização dos Espectrogramas gerados**: Plotagem de espectrogramas log-mel usando `matplotlib`.
    - **Reconstrução do Áudio**: Aplicação da transformada inversa (Griffin-Lim) para converter espectrogramas de volta ao domínio temporal, permitindo avaliação perceptual da qualidade do pré-processamento. Verificação de que o áudio reconstruído mantém inteligibilidade e características do original.

### 2. "Tokenização" do Espectrograma:

- Ao invés de tratar o espectrograma como uma imagem 2D, ele será dividido em janelas temporais fixas. Cada janela se tornará um "token" que representa um segmento do áudio.

- Cada "token" será codificado em um vetor de alta dimensão (embedding) que será a entrada para o modelo de mundo.

### 3. Arquitetura do Modelo de Mundo:

- A arquitetura será baseada no Recurrent State Space Model (RSSM), que define estados latentes estocásticos e determinísticos.

- O modelo de mundo será adaptado para processar a sequência de "tokens" de espectrograma. Ele consistirá em um codificador (encoder), um modelo de dinâmica latente (RSSM) e um decodificador (decoder). No entanto, o treinamento focará no objetivo de aprendizado contrastivo, eliminando a necessidade de reconstrução completa do espectrograma.

- O codificador transformará as janelas do espectrograma em um estado latente discreto, composto por variáveis categóricas.

- O RSSM será responsável por prever o próximo estado e recompensa (neste caso, a próxima janela de espectrograma) a partir do estado atual e de uma "ação" (pode ser um placeholder ou uma variável para representar a transição temporal).

### 4. Treinamento do Modelo:

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


## Base de referência
Este projeto é baseado em duas referências principais:

#### 1. Artigo Acadêmico: ["Dreaming V2: Reinforcement Learning with Discrete World Models without Reconstruction"](https://arxiv.org/pdf/2203.00494).

**Principais contribuições/inspirações**: O artigo apresenta o Dreaming V2, uma extensão colaborativa do DreamerV2 e Dreaming. Ele adota a representação discreta do DreamerV2 e um objetivo livre de reconstrução do Dreaming. O modelo de mundo é treinado usando um aprendizado contrastivo, que elimina a necessidade de reconstruir observações visuais complexas. Os autores demonstraram que esta abordagem alcança resultados de última geração em tarefas de braços robóticos em 3D.

#### 2. Bases de Código para Implementação:
[dreamer-torch](https://github.com/jsikyoon/dreamer-torch): Implementação em PyTorch se assemelha ao código original do DreamerV2, que foi escrito em TensorFlow. É uma referência valiosa para entender a estrutura e a lógica do modelo em um framework amplamente utilizado na comunidade de pesquisa. Os resultados demonstraram desempenho similar ao do modelo original em tarefas de controle de jogos.

[pydreamer](https://github.com/jurgisp/pydreamer): Outra reimplementação do DreamerV2 em PyTorch, que introduz algumas diferenças sutis e melhorias, como o uso de processos separados para o trainer e os workers do ambiente, permitindo que o GPU seja utilizado totalmente. Esta base de código serve como um ponto de partida para explorar abordagens ligeiramente diferentes e otimizações.


## Tecnologias e bibliotecas utilizadas
**Linguagem**: Python

**Frameworks de Deep Learning**: PyTorch (conforme as bases de código de referência)

**Processamento de Áudio**: Librosa, Torchaudio

**Manipulação de Dados**: NumPy, Pandas

**Visualização**: Matplotlib, TensorBoard

## Links para a Apresentação
- E1
    - Link para o [vídeo da apresentação](https://drive.google.com/file/d/1IFhNwxeS_8Gce3WTqXLOq8UJDLKJB7QQ/view?usp=sharing)
    - Link para os [slides da apresentação](https://www.canva.com/design/DAGzF_vtvEE/6c1_5Sw-mUuLSqV6HMjP9Q/edit?utm_content=DAGzF_vtvEE&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

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
