# deepRecModel
## require
- tf.version=1.14.0
- spark.version>=2.2.1
- scala.version>=2.11.12
## python model Lib
- models for recommendation system
- use tf.Estimator：分布式训练接口
- use dataset of Movielens-1M

#### implemented Models：
- recall model: LR, CF, MF

- rank model: 
    - FM系列：FM, FFM, deepFM, xDeepFM,
    - 多目标：ESSM，MMoE, PLE



## spark model Lib
- FM-spark
- FFM-spark
- deepFM-spark

##  to do list:
    - FM系列：NFM, AFM, BiFFM,  AoFFM, NeuralFFM
    - wide&deep：WDL
    - 双塔：DSSM
    - 序列模型：DIN, DIEN, BST
    - graph embedding: word2vec, item2vec, deepwalk, node2vec, EGES
    - DCN
    - NCF
    - PNN
    - FNN
    - MIND
    - GNN : graphSage, pinSage

