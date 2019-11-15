# 2ª Iteração
Uma forma de resolver o problema será teres mais dados. Isso podes conseguir com data augmentation (que não parece ser o que estás a fazer, apesar de teres uma secção com esse nome no readme)
  - interpretei mal o conceito de *data augmentation*; vou experimentar nos próximos testes

Também podes simplificar a rede. No Readme não percebo qual é a rede que estás a treinar. Usas a VGG16 ou Facenet para obter 64 features e acrescentas a label, mas depois treinas uma rede para te dar o vector final? Ou estás a treinar a VGG16 ou a Facenet com isto?
  - estou a treinar a VGG16 ou Facenet com a rede concatenada (que está a baixo):
  ![](https://i.ibb.co/bz9MSyk/final-model.png)
  - na imagem a cima, o "*embeddings_model*" é a rede que produz o vector de features, que pode ser a VGG16 ou a Facenet (**nota**: neste caso, a rede recebe imagens 224x224x3 e produz um vector de 64 features, mas no último teste está a receber imagens de 160x160x3 e a produzir um vector de 128 features)
  - os últimos testes foram feitos usando a Facenet (com a versão InceptionResNetV1), que produz um vector de 128 features (adicionei uma camada de normalização L2 no final da rede, tal como foi feito no paper da Facenet)
  - ou seja, neste caso, a rede concatenada produz um vector de 129 dimensões (128 features + 1 label), e treina o *embeddings model* (Facenet)
  - depois de treinar a rede concatenada, carrego os pesos apenas da rede dos embeddings (VGG16 ou Facenet) para uma VGG16 ou Facenet, e produzo o vector de features que eu quero obter, para testar na estrutura de dados métrica

Finalmente, há métodos de regularização eficazes, como dropout ou batch normalization, que podem ajudar no overfitting, mas estes podem ter efeitos indesejados no resultado
  - a Facenet já utiliza dropout e batch normalization (o código da criação da rede está [aqui](utils/models/facenet.py#L105-L217))

# 1ª Iteração
Descreveres os dados e pré-processamento, incluindo como dividiste em treino e validação, se fizeste data augmentation (e como) e a geração dos batches para treino:
  - **dados**:
  IMDB-WIKI (https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/), dataset com 500K+ imagens de rosto, construído com *web scraping* dos sites IMDB e Wikipedia, com anotações da idade; dividido em 2 datasets:
    - WIKI: 62,328 imagens de rosto (32,399 depois do pré-processamento);
    - IMDB: 460,723 imagens de rosto (*TODO* depois do pré-processamento)
    - cada dataset está associado a um ficheiro de metadados (wiki.mat e imdb.mat) com anotações da data de nascimento, data da fotografia, posição do rosto na imagem, entre outros (nota: o ficheiro imdb.mat está corrompido, mas os nomes dos ficheiros das imagens contêm a data de nascimento e data da fotografia, por isso foi assim que contornei o problema e extraí as idades das pessoas nas fotografias);
    - para cada dataset existe uma versão *cropped*, onde os rostos já estão "recortados"; a versão *cropped* foi o meu ponto de partida;
  - ***nota***: o dataset **IMDB** tem 2 problemas:
    - apesar de conter 460,723 imagens de rosto, só existem 20,284 celebridades diferentes, o que reduz a variação, afectando a generalização  
    - o dataset agrupa as celebridades; ou seja, existem ~25 imagens seguidas do rosto de Rowan Atkinson, depois ~25 do de Christian Bale, etc; o que acontece é que no meio de cada conjunto de ~25 imagens de rosto, existem entre 2-6 rostos que não correspondem à celebridade, resultando numa idade que não corresponde ao rosto na fotografia (suspeito que este problema tenha a ver com o algoritmo de *web scraping*); por este motivo, não utilizei este dataset para o critério da idade
  - **pré-processamento** (igual para o WIKI e IMDB, à exceção da extração da idade):
    1. removi as imagens corrompidas
    2. removi as imagens que não continham um rosto
    3. removi os outliers (18 <= idade <= 58)
    4. *data augmentation*
    5. extração dos labels para um ficheiro .pickle, em array (eg.: ages[ ] -> ages.pickle ou eigenvalues[ ] -> eigenvalues.pickle)
    6. renomeio os ficheiros de 0 ao numero total de imagens, de tal modo que a imagem <índice>.png corresponda ao label na posição índice do array criado no ponto 5. (eg.: para o critério da idade, o rosto na imagem 3541.png tem ages[3541] anos)
  - **data augmentation**:
    1. alinhei os rostos em relação aos olhos, utilizando a class *FaceAligner*, da library *imutils* (https://github.com/jrosebr1/imutils/blob/master/imutils/face_utils/facealigner.py)
    2. para cada imagem já alinhada e *cropped* pelos autores dos datasets, extraí o rosto utilizando a rede MTCNN (https://github.com/ipazc/mtcnn); esta extração estica também o rosto para o tamanho desejado (224x224 ou 160x160, dependendo da rede (VGG16 ou Facenet))
    eg. (1. | 2.):
    ![](https://i.ibb.co/HYHPfBR/0.jpg)
    ![](https://i.ibb.co/L5G8Lcm/1.jpg)
    ![](https://i.ibb.co/6NM4Mq0/2.jpg)
  - **como dividiste em treino e validação**: 
    - 90% treino
    - 5% validação
    - 5% teste
  - **geração dos batches para treino**:
    - implementei *data generators* do *Keras*, utilizados para os 3 conjuntos: treino, validação e teste;
    - cada *data generator* contém um array de indices, que serve para carregar imagens do dataset através do seu índice (eg.: <índice>.png, com label ages/eigenvalues[índice])
    - tenho 3 [data generators](utils/data/data_generators.py):
        - [AgeDG](utils/data/data_generators.py#L26-L81): critério de semelhança -> idade
        - [AgeIntervalDG](utils/data/data_generators.py#L84-L162): critério de semelhança -> intervalo de idade          
        - [EigenvaluesDG](utils/data/data_generators.py#L165-L220): critério de semelhança -> eigenvalue       

As arquitecturas que testaste e a função de loss que estás a usar (como a implementaste e como está encaixada no treino; nesta parte se calhar preciso que partilhes o código e digas quais as partes relevantes):
  - **arquitecturas que testaste**:
    - a arquitetura base consiste na concatenação de uma rede de *feature extraction* (eg. uma rede que gera embeddings, por exemplo VGG16 ou Facenet) e o label da imagem de input (eg. idade ou eigenvalue)
    eg. (imagem de input: 224x224, tamanho do embedding: 64, label: idade):
    ![](https://i.ibb.co/bz9MSyk/final-model.png)
    - na imagem a cima, testei utilizar como *embeddings_model* a VGG16 (https://neurohive.io/en/popular-networks/vgg16/) e a Facenet (https://arxiv.org/abs/1503.03832)
  - **função de loss que estás a usar**:
  estou a usar a *triplet loss function*
    - **como a implementaste**:
        - usei uma implementação do tensorflow,  (https://github.com/tensorflow/addons/blob/v0.6.0/tensorflow_addons/losses/triplet.py#L63-L131), que utiliza a estratégia *semihard triplet mining* ("semihard triplets: triplets where the negative is not closer to the anchor than the positive, but which still have positive loss: d ( a , p ) < d ( a , n ) < d ( a , p ) + m a r g i n" (https://omoindrot.github.io/triplet-loss)); no meu código está [aqui](utils/loss_functions/semihard_triplet_loss.py#L60-L141)
        - esta função utiliza 3 métodos auxiliares (cuja implementação foi também retirada do tensorflow):
            - [pairwise_distance](utils/loss_functions/distance_functions.py#L5-L41) : calcula a distância, euclideana ou manhattan, entre todos os triplos de cada *batch* 
            - [masked_minimum](utils/loss_functions/semihard_triplet_loss.py#L42-L57) : "Computes the axis wise minimum over chosen elements." (utilizada para obter os triplos mais próximos)
            - [masked_maximum](utils/loss_functions/semihard_triplet_loss.py#L23-L39) : "Computes the axis wise maximum over chosen elements." (utilizada para obter os triplos mais distantes)
    - **como está encaixada no treino**:
        - para cada batch
            1. são formados os triplos válidos (ancora, positivo, negativo)
            2. são descartados os que não correspondem à especificação de *semihard triplet*
            3. é calculada a *loss* da batch
    - ***nota sobre a implementação da semihard triplet loss***:
        - a implementação original do tensorflow recebe como argumentos (y_true, y_pred, margin=1.0), onde:
            - y_true: 1-D integer `Tensor` with shape [batch_size] of multiclass integer labels.
            - y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should be l2 normalized.
            - margin: Float, margin term in the loss definition.
        - a minha arquitetura produz um array que contém o embedding e o label associado (eg. tamanho do embedding: 64, label: idade -> array com 65 dimensões, onde array[:1] -> idade & array[1:] -> embedding); 
        - logo, tive de adicionar o seguinte código no [início da função](utils/loss_functions/semihard_triplet_loss.py#L60-L65):
        ```python
        # não existe um y_true; o array que contém o embedding e o label é o y_pred
        del y_true
        labels = y_pred[:, :1]
        embeddings = y_pred[:, 1:]
        del y_pred
        ```

Os parâmetros de treino (épocas, parâmetros do optimizador) e os resultados (os plots da loss no treino e na validação):
  - **parâmetros de treino**:
    - **épocas**: 
        - experimentei 5, 10, 15, 20, 25 
    - **parâmetros do optimizador**:
        - Adam Optimizer com 1^-4 learning rate
  - **resultados**:
    - só nos últimos testes é que passei a guardar a *val_loss*, porque os resultados nunca faziam muito sentido (a *val_loss* não descia em comparação com a *tra_loss*);
    - **plots da loss no treino e validação**: neste momento só tenho um plot da *val_loss*, com os seguintes parâmetros:
        - Criterion:			age
        - Triplet strategy:	adapted_semihard
        - CNN: facenet
        - Embedding size: 128
        - Number of epochs:	20
        - Batch size: 66
        - Shuffle: True
        - Faces aligned: True
        - Age scoped: True
        - Age interval: True
        - Uniformized: True
        - Training size 19404
        - Dataset size: 30060
        - Validation size: 1584
        - Testing size: 9072
        - Model name: es_128_e_20_bs_66_ts_19404_s_1_as_1_ar_1_ai_1_u_1_fa_1.h5 
        ![](https://i.ibb.co/s1M56Gh/val-loss.png)
   - ***notas***: 
      1. o nome do modelo é gerado de acordo com as seguintes regras:
          - es_< EMBEDDING SIZE >_
          - e_< NUMBER OF EPOCHS >_
          - bs_< BATCH SIZE >_
          - ts_< TRAINSIZE >_
          - s_< SHUFFLE (0/1) >_
          - as_< AGE SCOPED (0/1) >_
          - ar_< AGE RELAXED (0/1) >_
          - ai_< AGE INTERVAL (0: none/1: relaxed/2: 5in5/3: 10in10) >_
          - u_< UNIFORMIZED TRAINING (0/1) >
          - fa_< FACES ALIGNED (0/1) >.h5
        
        
      2. neste teste, a divisão do dataset em treino/validação/teste não correspondeu aos 90/5/5, pois o conjunto de treino está "uniformizado", e o valor máximo para o mesmo número de intervalos é 3234; ou seja, o conjunto de treino tem o mesmo número de imagens para cada intervalo de idades, para que, em cada batch, exista o mesmo número de imagens de cada intervalo (i.e.: training size: 19404, #intervalos: 6, #images de cada intervalo: 19404/6=3234, batch size: 66 -> cada batch contém 11 imagens de cada intervalo; esta técnica é utilizada na literatura para melhorar o treino ([Facenet](https://arxiv.org/pdf/1503.03832.pdf), secção 3.2 Triplet Selection: "*In our experiments we sample the training data such that around 40 faces are selected per identity per mini-batch.*")
        


Diz-me também que testes já fizeste ao código para garantir que está tudo a funcionar:
  - **rede**:
    - **geral**: 
        - antes de partir para o dataset dos rostos e as redes VGG16 e Facenet, testei a arquitetura (inspirada em: https://github.com/AdrianUng/keras-triplet-loss-mnist) no dataset MNIST, uma simples simples CNN como *embeddings_model* 
    - **triplet loss function**:
        - depois de treinar a rede, verifico no [tensorboard](https://projector.tensorflow.org/) se os embeddings do mesmo label estão perto uns dos outros; primeiro com o conjunto de teste, com o objectivo de perceber se a rede aprendeu alguma coisa (o que **não** se verificou...); depois de constatar que a rede não aprendeu nada, produzo embeddings para uma amostra do conjunto de treino para confirmar se a rede ajustou os pesos de acordo com a *triplet loss function* (o que se verificou)
  - **dados**:
    - **depois de pré-processar um dataset**:
        - iterei as imagens processadas a confirmar se o label era igual ao original
        - confirmei se tinham o tamanho pretendido (224x224 ou 160x160)
    - **geração dos batches**:
        - o ficheiro [tester.py](tests/tester.py) contém os testes feitos aos *Data Generators*; no fundo, simulo uma época, verifico se o número de batches produzido pelo *data generator* é igual ao esperado (*set_size // batch_size*), e no fim de cada época, baralho os índices
