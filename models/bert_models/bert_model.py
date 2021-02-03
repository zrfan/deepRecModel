import collections
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp

def checkModel(path):
    var = chkp.print_tensors_in_checkpoint_file(path+"/model.ckpt", tensor_name="", all_tensors=True)

    print("all var below:")
    print(var)

if __name__ == "__main__":
    checkModel("../data/pretrain/")

