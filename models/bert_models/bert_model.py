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
            # var = graph.get_tensor_by_name(k+":0")
            print("get variable:", k)

        print("all var len=", len(tensor_name_list))
        tokens = [1, 4, 7, 23, 5, 3]   # ["[CLS]", "我", "爱", "中", "国", "[SEP]"]
        seg_ids = [0, 0, 0, 0, 0, 0]
        mask = [1] * len(tokens)
        max_seq_length = 128
        while len(seg_ids) < max_seq_length:
            tokens.append(0)
            seg_ids.append(0)
            mask.append(0)
        # model inputs: input_ids , input_mask, segment_ids, label_id
        input_ids = graph.get_tensor_by_name("input_ids:0")
        input_mask = graph.get_tensor_by_name("input_mask:0")
        segment_ids = graph.get_tensor_by_name("segment_ids:0")

        output = graph.get_tensor_by_name("bert/pooler/dense/Tanh:0")
        result = sess.run(output, feed_dict={input_ids: [tokens], input_mask: [mask], segment_ids: seg_ids})
        print("result=", result)

        constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def)

        with tf.gfile.FastGFile(path+'/test_model.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())



if __name__ == "__main__":
    path = "../../data/pretrain_model/"
    # checkModelTensor(path)
    checkModelGraph(path)

