import collections
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp

def checkModel(path):
    print("all var below:")
    chkp.print_tensors_in_checkpoint_file(path+"/albert_model/albert_model.ckpt", tensor_name="", all_tensors=True)



if __name__ == "__main__":
    checkModel("../../data/pretrain_model/")

