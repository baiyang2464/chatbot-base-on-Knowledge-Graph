
# coding: utf-8
import data_parser
import tensorflow as tf
import numpy as np
import json

corpus_path = "../data/cbowData/classifyDocument.txt"
embedding_word_path = corpus_path+".ebd"
vacb_path = corpus_path+".vab"
data = data_parser.TextLoader(corpus_path,batch_size=300)

with tf.device('/gpu:1'):
    embedding_word_length = 200
    learning_rate =0.0001
    #输入
    input_x  = tf.placeholder(dtype=tf.float32,shape=[None,data.vacb_size],name="inputx")
    input_y  = tf.placeholder(dtype=tf.float32,shape=[None,data.vacb_size],name='inputy')

    try:
        last_word_embeddings = np.load(embedding_word_path+".npy")
    except:
        print("训练新模型")
        W1 = tf.Variable(name="embedding_word",initial_value=tf.truncated_normal(shape=[data.vacb_size,embedding_word_length],stddev=1.0/(data.vacb_size)))
    else:
        if last_word_embeddings.shape !=(data.vacb_size,embedding_word_length):
            print("新模型shape改变，重新训练模型")
            W1 = tf.Variable(name="embedding_word",initial_value=tf.truncated_normal(shape=[data.vacb_size,embedding_word_length],stddev=1.0/(data.vacb_size)))
        else:
            print("加载已有模型继续训练")
            W1 = last_word_embeddings

    #W1其实就是 词向量矩阵
    W2 = tf.Variable(tf.truncated_normal(shape=[embedding_word_length,data.vacb_size],stddev=1.0/data.vacb_size))
    #计算过程
    #累加所有的X,然后取平均值，再去乘以词向量矩阵;其效果就是相当于,选择出所有的上下文的词向量,然后取平均
    hidden = tf.matmul(input_x,W1)
    output = tf.matmul(hidden,W2) #batch_size * vacb_size的大小
    output_softmax = tf.nn.softmax(output)
    #取出中心词的那个概率值,因为iinput_y是一个one-hot向量,左乘一个列向量,就是将这个列的第i行取出
    output_y = tf.matmul(input_y,output_softmax,transpose_b=True)
    loss = tf.reduce_sum(- tf.log(output_y)) #将batch里面的output_y累加起来
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    max_epoch = 2
    max_iteration =data.number_batch
    print("max_iteration=%d"%(max_iteration))
    lastloss = -1
    for epoch in range(max_epoch):
        for iteraion in range(1,max_iteration):
            _x,_y = data.next_batch()
            #生成本次的输入
            x =[]
            y =[]
            for i in range(0,len(_x)):
                #将Context*2 -1 个向量,求和取平均为一个向量,用于将来和词向量矩阵相乘
                vec = np.zeros(shape=[data.vacb_size]) #vacb_size为词向量语料中无重复单词的数目
                for j in range(0,len(_x[i])):
                    vec[ _x[i][j] ] += 1
                vec /= len(_x[i])
                x.append(vec)
                y_vec = np.zeros(shape=[data.vacb_size])
                y_vec[_y[i]] = 1
                y.append(y_vec)
            _loss,_ = sess.run([loss,train_op],feed_dict={input_x:x,input_y:y})
            if (iteraion % 100 )==0 :
                print("loss:%.2f  iteraion:%d  epoch:%d"%(_loss,iteraion+epoch*max_iteration,epoch))
                if lastloss < 0:
                    lastloss = _loss
                elif lastloss>0:
                    if lastloss-_loss >100 or _loss< 1000 and lastloss >_loss:
                        print("保存词向量模型..")
                        lastloss = _loss
                        #保存词向量
                        _W1 = sess.run(W1,feed_dict={input_x:[np.zeros([data.vacb_size])],input_y:[np.zeros([data.vacb_size])]})
                        #每一行就是对应词的词向量
                        np.save(embedding_word_path,_W1)
                        with open(vacb_path,"w",encoding='utf8') as fp:
                            json.dump(data.inverseV,fp)
    print("Some TEST:")
    print("<START>:",_W1[data.V['<START>']])
    print("<END>:",_W1[data.V['<END>']])
    print("<UNK>:",_W1[data.V['<UNK>']])
    print("EDN TRAIN...")

