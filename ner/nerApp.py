
# coding: utf-8

import  tensorflow as tf
from  tensorflow.contrib import  crf
import  random
from  utils import *
import logging
import datetime

tf.reset_default_graph()
with tf.device('/gpu:1'):
    #参数
    batch_size=1
    dataGen = DATAPROCESS(train_data_path="../data/nerData/train_cutword_data.txt",
                              train_label_path="../data/nerData/label_cutword_data.txt",
                              test_data_path="../data/nerData/test_data.txt",
                              test_label_path="../data/nerData/test_label.txt",
                              word_embedings_path="../data/cbowData/document.txt.ebd.npy",
                              vocb_path="../data/cbowData/document.txt.vab",
                              batch_size=batch_size
                            )
    #模型超参数
    tag_nums =len(dataGen.state)    #标签数目
    hidden_nums = 650                #bi-lstm的隐藏层单元数目
    learning_rate = 0.00075          #学习速率
    sentence_len = dataGen.sentence_length #句子长度,输入到网络的序列长度
    frame_size = dataGen.embedding_length #句子里面每个词的词向量长度

    #网络的变量
    word_embeddings =  tf.Variable(initial_value=dataGen.word_embeddings,trainable=True) #参与训练
    #输入占位符
    input_x = tf.placeholder(dtype=tf.int32,shape=[None,None],name='input_word_id')#输入词的id
    input_y = tf.placeholder(dtype=tf.int32,shape=[None,sentence_len],name='input_labels')
    sequence_lengths=tf.placeholder(dtype=tf.int32,shape=[None],name='sequence_lengths_vector')
    #
    with tf.name_scope('projection'):
        #投影层,先将输入的词投影成相应的词向量
        word_id = input_x
        word_vectors = tf.nn.embedding_lookup(word_embeddings,ids=word_id,name='word_vectors')
        #word_vectors = tf.nn.dropout(word_vectors,0.8)
    with tf.name_scope('bi-lstm'):

        #labels = tf.reshape(input_y,shape=[-1,sentence_len],name='labels')
        #labels = tf.reshape(input_y,shape=[-1,tag_nums],name='labels')
        labels = tf.reshape(input_y,shape=[batch_size,sentence_len],name='labels')
        fw_lstm_cell =tf.nn.rnn_cell.LSTMCell(hidden_nums)
        bw_lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_nums)
        #双向传播
        output,_state = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell,bw_lstm_cell,inputs=word_vectors,sequence_length=sequence_lengths,dtype=tf.float32)
        fw_output = output[0]#[batch_size,sentence_len,hidden_nums]
        bw_output =output[1]#[batch_size,sentence_len,hidden_nums]
        V1=tf.get_variable('V1',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=[hidden_nums,hidden_nums])
        V2=tf.get_variable('V2',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=[hidden_nums,hidden_nums])
        fw_output = tf.reshape(tf.matmul(tf.reshape(fw_output,[-1,hidden_nums],name='Lai') , V1),shape=tf.shape(output[0]))
        bw_output = tf.reshape(tf.matmul( tf.reshape(bw_output,[-1,hidden_nums],name='Rai') , V2),shape=tf.shape(output[1]))
        contact = tf.concat([fw_output,bw_output],-1,name='bi_lstm_concat')#[batch_size,sentence_len,2*hidden_nums]
        contact = tf.nn.dropout(contact,0.9)
        s=tf.shape(contact)
        contact_reshape=tf.reshape(contact,shape=[-1,2*hidden_nums],name='contact')
        W=tf.get_variable('W',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=[2*hidden_nums,tag_nums],trainable=True)
        b=tf.get_variable('b',initializer=tf.zeros(shape=[tag_nums]))
        p=tf.nn.relu(tf.matmul(contact_reshape,W)+b)
        #logit= tf.reshape(p,shape=[-1,s[1],tag_nums],name='omit_matrix')
        #logit= tf.reshape(p,shape=[-1,s[1],sentence_len],name='omit_matrix')
        logit= tf.reshape(p,shape=[batch_size,sentence_len,tag_nums],name='omit_matrix') 

    with tf.name_scope("crf") :
        log_likelihood,transition_matrix=crf.crf_log_likelihood(logit,labels,sequence_lengths=sequence_lengths)
        cost = -tf.reduce_mean(log_likelihood)
        crf_labels,_=crf.crf_decode(logit,transition_matrix,sequence_length=sequence_lengths) #返回的第一个值:decode_tags: A [batch_size, max_seq_len]
    with tf.name_scope("train-op"):
        global_step = tf.Variable(0,name='global_step',trainable=False)
        optim = tf.train.AdamOptimizer(learning_rate)
        #train_op=optim.minimize(cost)
        grads_and_vars = optim.compute_gradients(cost)
        grads_and_vars = [[tf.clip_by_value(g,-5,5),v] for g,v in grads_and_vars]
        train_op = optim.apply_gradients(grads_and_vars,global_step)

    #载入模型如果有参数的话
    model_name="bilstm-crf.models"
    model_checkpoint_path ="./nerModel/"
    saver = tf.train.Saver(max_to_keep=1)

    display_step = len(dataGen.train_batches)
    epoch_nums = 3 #迭代的数据轮数
    max_batch = len(dataGen.train_batches)*epoch_nums

    step= 0 
    last_f1 = -1 #用于保存上一次迭代时候的损失
    debug = False

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
    ckpt = tf.train.get_checkpoint_state(model_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        logging.info("restore from history model...")
        saver.restore(sess,ckpt.model_checkpoint_path)
        logging.info("model loading successful")
        #把输入的字分隔开，
        text = "application"
        while(text!="" and text!=" "):
            text=input("请输入一句话：")
            if text == "quit" or text=="" or text == " ":break
            data_line,data_x,efficient_sequence_length=dataGen.handleInputData(text)
            if debug:
                print(np.array(data_x).shape)
                print(np.array(efficient_sequence_length).shape)
            predict_labels = sess.run([crf_labels],feed_dict={input_x:data_x,sequence_lengths:efficient_sequence_length})
            #predict_labels是三维的[1,1,25]，第1维包含了一个矩阵
            lable_line =[]
            sentence_len_sum=0
            if debug:
                print(type(predict_labels))
                print(np.array(predict_labels).shape)
            for idx in range(len(predict_labels[0])):
                _label = predict_labels[0][idx].reshape(1,-1)
                lable_line += list(_label[0])
                sentence_len_sum = efficient_sequence_length[idx]
            for idx in range(sentence_len_sum):
                print("%s : %s"%(data_line[idx],lable_line[idx]))
    else:
        logging.warning("train a new models.")


