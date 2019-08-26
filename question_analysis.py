
# coding: utf-8

import tensorflow as tf
from classifyApp import classifyApplication
from nerApp import nerAppication
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加
log_device_placement = True  # 是否打印设备分配日志
allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
session_conf = tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement= allow_soft_placement,log_device_placement= log_device_placement)

class question_ays:
    def __init__(self,device='/cpu:0'):
        self.g1 = tf.Graph() #为每个类(实例)单独创建一个graph
        self.g2 = tf.Graph()
        self.device = device
        self.id2state={0:'O',
                    1:'B-dis',2:'I-dis',3:'E-dis',
                    4:'B-sym',5:'I-sym',6:'E-sym',
                    7:'B-dru',8:'I-dru',9:'E-dru',
                    10:'S-dis',11:'S-sym',12:'S-dru'}
        self.sess_ner =  tf.Session(graph = self.g1,config=session_conf)
        self.sess_classify = tf.Session(graph = self.g2,config=session_conf)
        
        self.classifyApp = classifyApplication(self.sess_classify,device)
        
        self.nerApp = nerAppication(self.sess_ner,device)
        
        self.state2entityType={'dis':'disease','sym':'symptom','dru':'drug'}
        self.label2id={"disease_symptom":0,"symptom_curway":1,"symptom_disease":2,"disease_drug":3,
                       "drug_disease":4,"disease_check":5,"disease_prevent":6,
                       "disease_lasttime":7,"disease_cureway":8}
        self.id2label ={0:"disease_symptom",1:"symptom_curway",2:"symptom_disease",3:"disease_drug",
                       4:"drug_disease",5:"disease_check",6:"disease_prevent",
                       7:"disease_lasttime",8:"disease_cureway"}
    def analysis(self,text):
        res = {}
        args={}
        question_types=[]
        data_line,lable_line,efficient_sequence_length =self.nerApp.questionNer(self.sess_ner,text)  
        for idx in range(len(data_line)):
            middle_question= []
            _entity = ''
            for each in range(efficient_sequence_length[idx]):
                middle_question.append(data_line[idx][each])
                _entityType = self.id2state[int(lable_line[idx][each])]
                if _entityType[0]=='B' or  _entityType[0]=='I':
                    _entity+= data_line[idx][each]
                elif _entityType[0]=='E' or _entityType[0]=='S':
                    _entity+= data_line[idx][each]
                    _entityType_short =  _entityType[-3:] 
                    middle_question.append(self.state2entityType[_entityType_short])
                    if _entity not in args:
                        args.setdefault(_entity,[self.state2entityType[_entityType_short]])
                    else: args[_entity].append(self.state2entityType[_entityType_short])
                    _entity=''
                else:
                    _entity=''
            question_text = ''.join(middle_question)
            _classify_idx = self.classifyApp.questionClassify(self.sess_classify ,question_text)
            _classify_label = self.id2label[_classify_idx[0]]
            question_types.append(_classify_label)
        res['args'] = args
        res['question_types']=question_types
        return res 
        
if __name__=="__main__":
    ques = question_ays()
    text="我发烧流鼻涕怎么治疗"
    while(text!="" and text!=" "):
        text=input("请输入一句话：")
        if text == "quit" or text=="" or text == " ":break
        res=ques.analysis(text)
        print(res) 
    
