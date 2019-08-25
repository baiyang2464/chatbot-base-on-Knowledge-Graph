
# coding: utf-8

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from classifyUtils import data_process
from text_cnn import TextCNN
import math
from tensorflow.contrib import learn
import jieba 

#tf.reset_default_graph()

class classifyApplication:
    def __init__(self,sess,device='/gpu:1'):
        with sess.as_default():
            with sess.graph.as_default():
                self.word_embedings_path="./data_ai/cbowData/classifyDocument.txt.ebd.npy"
                self.vocb_path = "./data_ai/cbowData/classifyDocument.txt.vab"
                self.model_path="./data_ai/classifyModel"
                self.num_classes = 9
                self.max_sentence_len = 20
                self.embedding_dim = 200
                self.filter_sizes="2,3,4"
                self.dropout_keep_prob=1.0
                self.l2_reg_lambda=0.0
                self.num_filters=128 
                self.num_checkpoints =1 

                self.data_helpers = data_process(
                                                train_data_path="",
                                                word_embedings_path=self.word_embedings_path,
                                                vocb_path=self.vocb_path,
                                                num_classes=self.num_classes,
                                                max_document_length = self.max_sentence_len)
                self.data_helpers.load_wordebedding()
                self.cnn = TextCNN(
                                    w2v_model= self.data_helpers.word_embeddings,
                                    sequence_length=self.max_sentence_len,
                                    num_classes=self.num_classes,
                                    embedding_size= self.embedding_dim,
                                    filter_sizes=list(map(int,  self.filter_sizes.split(","))),
                                    num_filters= self.num_filters,
                                    l2_reg_lambda= self.l2_reg_lambda,
                                    device = device
                                    )
                self.saver = tf.train.Saver(max_to_keep= self.num_checkpoints)
                ckpt = tf.train.get_checkpoint_state(self.model_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.saver.restore(sess, ckpt.model_checkpoint_path)
                    print("restore from history model.")
                else:
                    print("there is no classify model.")
        
        
    def classifyApp(self,sess):
        with sess.as_default():
            with sess.graph.as_default():
                text="application"
                while(text!="" and text!=" "):
                    text=input("请输入一句话：")
                    if text == "quit" or text=="" or text == " ":break
                    text = text.strip()
                    seg_list=list(jieba.cut(text)) 
                    x_data = self.data_helpers.handle_input(' '.join(seg_list))
                    feed_dict = {self.cnn.input_x: x_data,self.cnn.dropout_keep_prob: self.dropout_keep_prob}
                    _predic = sess.run([self.cnn.predictions],feed_dict)
                    print("%s is %d"%(text,_predic[0]))
    def questionClassify(self,sess,text):
        with sess.as_default():
            with sess.graph.as_default():
                text = text.strip()
                seg_list=list(jieba.cut(text)) 
                x_data = self.data_helpers.handle_input(' '.join(seg_list))
                feed_dict = {self.cnn.input_x: x_data,self.cnn.dropout_keep_prob: self.dropout_keep_prob}
                _predic = sess.run([self.cnn.predictions],feed_dict)
                return _predic[0]

if __name__ == "__main__":
    graph = tf.Graph()
    # 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加
    log_device_placement = True  # 是否打印设备分配日志
    allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    session_conf = tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement= allow_soft_placement,log_device_placement= log_device_placement)
    
    sess =  tf.Session(graph=graph,config=session_conf)
    classifyApp =classifyApplication(sess)
    classifyApp.classifyApp(sess) 
