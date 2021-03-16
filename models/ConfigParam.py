# -*- coding: utf-8 -*

class ConfigParam:
    # params = {"embedding_size": 6, "feature_size": 0, "field_size": 0, "batch_size": 64, "learning_rate": 0.001,"epochs":200,
    #               "optimizer": "adam", "data_path": "../data/ml-1m/", "model_dir": "../data/model/essm/", "hidden_units":[8]}
    def __init__(self, params):
        self.param = params
        # self.embedding_size, self.feature_size = params["embedding_size"], params["feature_size"]
        # self.field_size, self.batch_size, self.learning_rate = params["field_size"], params["batch_size"], params["learning_rate"]
        # self.epochs, self.optimizer, self.data_path = params["epochs"], params["optimizer"], params["data_path"]
        # self.model_dir, self.hidden_units = params["model_dir"], params["hidden_units"]
        # if "experts_num" in params.keys() and "experts_units" in params.keys():
        #     self.experts_num, self.experts_units = params["experts_num"], params["experts_units"]
        # if "label1_weight" in params.keys() and "label2_weight" in params.keys():
        #     self.label1_weight, self.label2_weight = params["label1_weight"], params["label2_weight"]
        for key in params.keys():
            if not hasattr(self, key):
                setattr(self, key, params[key])
