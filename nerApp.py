
# coding: utf-8

import  tensorflow as tf
from  tensorflow.contrib import  crf
import  random
from  nerUtils import *
import logging
import datetime
from BiLSTM_CRF import BiLSTM_CRF

debug = False
batch_size =100
class nerAppication:
    #参数
    def __init__(self,sess,device='/gpu:1'):
        with sess.as_default():
            with sess.graph.as_default():
                self.dataGen = DATAPROCESS(train_data_path="./data_ai/nerData/train_cutword_data.txt",
                                          train_label_path="./data_ai/nerData/label_cutword_data.txt",
                                          test_data_path="./data_ai/nerData/test_data.txt",
                                          test_label_path="./data_ai/nerData/test_label.txt",
                                          word_embedings_path="./data_ai/cbowData/document.txt.ebd.npy",
                                          vocb_path="./data_ai/cbowData/document.txt.vab",
                                          batch_size=batch_size
                                        )
                self.dataGen.load_wordebedding()
                self.tag_nums =13  # 标签数目
                self.hidden_nums = 650  # bi-lstm的隐藏层单元数目
                self.sentence_len = self.dataGen.sentence_length # 句子长度,输入到网络的序列长度
                self.model_checkpoint_path ="./data_ai/nerModel/"
                self.model = BiLSTM_CRF(
                                        batch_size = batch_size,
                                        tag_nums=self.tag_nums,
                                        hidden_nums = self.hidden_nums,
                                        sentence_len = self.sentence_len,
                                        word_embeddings=self.dataGen.word_embeddings,
                                        device = device
                                        )
                self.saver = tf.train.Saver(max_to_keep=1)
                ckpt = tf.train.get_checkpoint_state(self.model_checkpoint_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.saver.restore(sess,ckpt.model_checkpoint_path)
                    logging.info("model loading successful")
  
    def nerApp(self,sess):
        with sess.as_default():
            with sess.graph.as_default():
                text = "application"
                while(text!="" and text!=" "):
                    text=input("请输入一句话：")
                    if text == "quit" or text=="" or text == " ":break
                    data_line,data_x,efficient_sequence_length=self.dataGen.handleInputData(text)
                    if debug:
                        print(np.array(data_x).shape)
                        print(data_x)
                        print(np.array(efficient_sequence_length).shape)
                    feed_dict={self.model.input_x:data_x, 
                               self.model.sequence_lengths:efficient_sequence_length,
                               self.model.dropout_keep_prob:1 
                              }
                    predict_labels = sess.run([self.model.crf_labels],feed_dict)#predict_labels是三维的[1,1,25]，第1维包含了一个矩阵
                    lable_line =[]
                    if debug:
                        print(type(predict_labels))
                        print(predict_labels)
                        print(np.array(predict_labels).shape)
                    for idx in range(len(predict_labels[0])):
                        _label = predict_labels[0][idx].reshape(1,-1)
                        lable_line.append( list(_label[0]))
                    for idx in range(len(data_line)):
                        for each in range(efficient_sequence_length[idx]):
                            print("%s:%s"%(data_line[idx][each],lable_line[idx][each]),end="  ")
                        print('\n')
                        
    def questionNer(self,sess,text):
        with sess.as_default():
            with sess.graph.as_default():
                if text== " ":
                    print("文本为空，错误")
                    return
                data_line,data_x,efficient_sequence_length=self.dataGen.handleInputData(text)

                feed_dict={self.model.input_x:data_x, 
                           self.model.sequence_lengths:efficient_sequence_length,
                           self.model.dropout_keep_prob:1 }
                predict_labels = sess.run([self.model.crf_labels],feed_dict)#predict_labels是三维的[1,1,25]，第1维包含了一个矩阵
                lable_line =[]
                for idx in range(len(predict_labels[0])):
                    _label = predict_labels[0][idx].reshape(1,-1)
                    lable_line.append( list(_label[0]))
                return data_line,lable_line,efficient_sequence_length

if __name__=="__main__":
    
    graph = tf.Graph()
    log_device_placement = True  # 是否打印设备分配日志
    allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    session_conf = tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement= allow_soft_placement,log_device_placement= log_device_placement)
    
    sess =  tf.Session(graph=graph,config=session_conf)
    app = nerAppication(sess) 
    
    text="我发烧流鼻涕怎么办"
    while(text!="" and text!=" "):
        text=input("请输入一句话：")
        if text == "quit" or text=="" or text == " ":break
        data_line,lable_line,efficient_sequence_length = app.questionNer(sess,text) 
        for idx in range(len(data_line)):
            for each in range(efficient_sequence_length[idx]):
                print("%s:%s"%(data_line[idx][each],lable_line[idx][each]),end="  ")
            print('\n')
        

