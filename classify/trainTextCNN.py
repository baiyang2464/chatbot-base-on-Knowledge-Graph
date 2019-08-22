
# coding: utf-8
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_input_helper
from text_cnn import TextCNN
import math
from tensorflow.contrib import learn

tf.reset_default_graph()
# Parameters
# Model Hyperparameters
embedding_dim = 200  # Dimensionality of character embedding default 200
filter_sizes="2,3,4" # Comma-separated filter sizes default: '3,4,5'
num_filters=128     # Number of filters per filter size defaul 128
dropout_keep_prob=0.5
l2_reg_lambda=0.0    # L2 regularization lambda
                     # 注：l2_reg_lambda的选取：设置为0，先确定一个比较好的learning rate。然后固定该learning rate，
                     # 给λ一个值（比如1.0），然后根据validation accuracy，将λ增大或者减小10倍（增减10倍是粗调节，
                     # 当你确定了λ的合适的数量级后，比如λ = 0.01,再进一步地细调节，比如调节为0.02，0.03，0.009之类。）
max_sentence_len = 20
num_classes =9
learn_rate = 1e-4

# Training parameters
batch_size = 300 
num_epochs = 100 
log_every = 100
evaluate_every =500
num_checkpoints =1 
last_acc = -1

# Misc Parameters
allow_soft_placement = True # Allow device soft device placement
log_device_placement = True # Log placement of ops on devices

def trainBegin(train_data_path,word_embedings_path,vocb_path,model_path,num_classes=9):
    # Training
    # load data and word2vec_model
    data_helpers = data_input_helper.data_process(
            train_data_path=train_data_path,
            word_embedings_path=word_embedings_path,
            vocb_path=vocb_path,
            num_classes=num_classes
            max_document_length = max_sentence_len
            )
    data_helpers.load_wordebedding()
    x_train, x_dev, y_train, y_dev = data_helpers.load_data()
    
    # 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加
    log_device_placement = True  # 是否打印设备分配日志
    allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    session_conf = tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement= allow_soft_placement,log_device_placement= log_device_placement)
    
    cnn = TextCNN(
            w2v_model= data_helpers.word_embeddings,
            sequence_length=max_sentence_len,
            num_classes=num_classes,
            embedding_size= embedding_dim,
            filter_sizes=list(map(int,  filter_sizes.split(","))),
            num_filters= num_filters,
            l2_reg_lambda= l2_reg_lambda)
    print("建计算图完毕...") 
    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    model_name = os.path.join(model_path, "model")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep= num_checkpoints)
    
    with tf.Session(config=session_conf) as sess:
        # Initialize all variables
        print("加载或新建模型...")
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore from history model.")
        else:
            sess.run(tf.global_variables_initializer())
            print("train a new model.")

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {cnn.input_x: x_batch,cnn.input_y: y_batch,cnn.dropout_keep_prob:  dropout_keep_prob}
            _,step,_loss,_accuracy = sess.run([train_op, global_step, cnn.loss,cnn.accuracy],feed_dict) 
            time_str = datetime.datetime.now().strftime('%H:%M:%S')
            if step %  log_every == 0:
                print("time {}: step {}, loss {:.4f}, accuracy {:.4f}".format(time_str, step, _loss, _accuracy))

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)),  batch_size,  num_epochs )
        
        def dev_test():
            """
            Evaluates model on a dev set
            """
            global last_acc
            batches_dev = data_helpers.batch_iter(list(zip(x_dev, y_dev)),  batch_size, 1)
            accuracy = 0
            count = 0 
            for batch_dev in batches_dev:
                x_batch_dev, y_batch_dev = zip(*batch_dev)
                feed_dict = {cnn.input_x: x_batch_dev,cnn.input_y: y_batch_dev,cnn.dropout_keep_prob: 1.0}
                _predic,_accuracy = sess.run([cnn.predictions,cnn.accuracy],feed_dict)
                accuracy +=_accuracy 
                count +=1
            if count>0: accuracy/=count
            time_str = datetime.datetime.now().strftime('%H:%M:%S')
            print("time {}: accuracy on valid dataset {:.4f} ".format(time_str, accuracy))
            if last_acc <0: 
                last_acc = accuracy
            elif last_acc>0 and  accuracy >= last_acc:
                last_acc = accuracy
                path = saver.save(sess, model_name, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
                
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            #Training loop. For each batch...
            if current_step %  evaluate_every == 0:
                dev_test()
                    
if __name__ == "__main__":
    trainBegin(train_data_path="../data/classifyData/train_data.txt",
          word_embedings_path="../data/cbowData/classifyDocument.txt.ebd.npy",
          vocb_path="../data/cbowData/classifyDocument.txt.vab",
          model_path="./classifyModel",
          )
