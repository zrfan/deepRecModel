import collections
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp

def checkModelTensor(path):
    print("all var below:")
    chkp.print_tensors_in_checkpoint_file(path+"/albert_model/albert_model.ckpt", tensor_name="", all_tensors=True)

#     cls/predictions/transform/dense/bias

    chkp.print_tensors_in_checkpoint_file(path + "/albert_model/albert_model.ckpt", tensor_name="cls/predictions/transform/dense/bias", all_tensors=True)

def checkModelGraph(path):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 加载模型结构
        saver = tf.train.import_meta_graph(path+"/albert_model/albert_model.ckpt.meta")
        # 载入模型参数
        saver.restore(sess, path + "/albert_model/albert_model.ckpt")
        graph = tf.get_default_graph()  # 获取当前图，为了后续训练时恢复变量
        tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node] # 得到当前图中所有变量的名称
        for k in tensor_name_list:
            print("get variable:", k)


if __name__ == "__main__":
    path = "../../data/pretrain_model/"
    # checkModelTensor(path)
    checkModelGraph(path)

