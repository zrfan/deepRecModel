# -*- coding: utf-8 -*
# tf1.14
import tensorflow as tf
import abc

class BaseEstimatorModel(object):
    def __init__(self):
        pass
    @abc.abstractmethod
    def model_fn(self):
        raise NotImplementedError
    def model_estimator(self, params):
        tf.reset_default_graph()
        session_config = tf.ConfigProto(log_device_placement=True, device_count={'GPU': 0})
        session_config.gpu_options.per_process_gpu_memory_fraction = 0.8
        config = tf.estimator.RunConfig(keep_checkpoint_max=2, log_step_count_steps=500, save_summary_steps=50,
                                        save_checkpoints_steps=50000).replace(session_config=session_config)
        model = tf.estimator.Estimator(model_fn=self.model_fn, model_dir=params["model_dir"], params=params, config=config)
        return model