#coding=utf-8
import tensorflow as tf
import numpy
from baselines.classifier.src.model.basic_cnn import ConvNet


class ClCNN(object):
    def __init__(self, sess, model_path, graph):
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        self.sess = sess
        self.graph = graph
        #self.graph_path = graph_path
        self.model_path = model_path
        with self.graph.as_default():
            self.model = ConvNet(n_channel=1, n_classes=4, image_height=168, image_width=84) 
            dict_cnn = {}
            for variable in tf.global_variables():
                if(not variable.name.startswith('a2c_model')):
                    dict_cnn[variable.name.split(':')[0]] = variable
            self.saver = tf.train.Saver(dict_cnn, write_version=tf.train.SaverDef.V2)
            self.saver.restore(self.sess, self.model_path)       
            
    def predict(self, input_image): 
        with self.sess.as_default():
            #return self.sess.run(self.output, {self.input:input_image, self.keep_prob:1})
            [logit] = self.sess.run(
                            fetches=[self.model.logit], 
                            feed_dict={self.model.images:input_image, self.model.keep_prob:1.0})
            return logit