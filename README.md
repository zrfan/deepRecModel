# deepRecModel
## require
- tf.version=1.14.0
## python model Lib
- models for recommendation system
- use tf.Estimator：分布式训练接口
- use dataset of Movielens-1M

#### implemented Models：
- recall model: LR, CF, MF

- rank model: 
    - FM系列：FM, FFM, deepFM, xDeepFM,
    - 多目标：ESSM，MMoE, PLE

- to do list:
    - FM系列：BiFFM,  AoFFM, NeuralFFM
    - wide&deep：WDL
    - 双塔：DSSM
    - 序列模型：DIN, DIEN, BST
    - DCN
    - NFM
    - NCF
    - AFM
    - PNN
    - FNN
    - PLE
    - graph embedding: word2vec, item2vec, deepwalk, node2vec, EGES
    - GNN : graphSage, pinSage

## spark model Lib
- FM-spark
- FFM-spark
- deepFM-spark

