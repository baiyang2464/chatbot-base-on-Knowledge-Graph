#coding:utf-8
import  json
import  jieba
import  copy
import  numpy as np
import  random


class  DATAPROCESS:
    def __init__(self,train_data_path,train_label_path,test_data_path,test_label_path,word_embedings_path,vocb_path,seperate_rate=0.1,batch_size=100):
        self.train_data_path =train_data_path
        self.train_label_path =train_label_path
        self.test_data_path = test_data_path
        self.test_label_path = test_label_path
        self.word_embedding_path = word_embedings_path
        self.vocb_path  = vocb_path
        self.seperate_rate =seperate_rate
        self.batch_size = batch_size
        self.sentence_length = 25
        self.state={'O':0,
            'B-dis':1,'I-dis':2,'E-dis':3,
            'B-sym':4,'I-sym':5,'E-sym':6,
            'B-dru':7,'I-dru':8,'E-dru':9,
            'S-dis':10,'S-sym':11,'S-dru':12}
        self.id2state={0:'O',
            1:'B-dis',2:'I-dis',3:'E-dis',
            4:'B-sym',5:'I-sym',6:'E-sym',
            7:'B-dru',8:'I-dru',9:'E-dru',
            10:'S-dis',11:'S-sym',12:'S-dru'}

        #data structure to build
        self.train_data_raw=[]
        self.train_label_raw =[]
        self.valid_data_raw=[]
        self.valid_label_raw = []

        self.test_data_raw =[]
        self.test_label_raw =[]

        self.word_embeddings=None
        self.id2word=None
        self.word2id=None
        self.embedding_length =0

        self.last_batch=0
    def load_wordebedding(self):
        self.word_embeddings=np.load(self.word_embedding_path)
        self.embedding_length = np.shape(self.word_embeddings)[-1]
        with open(self.vocb_path,encoding="utf8") as fp:
            self.id2word = json.load(fp)
        self.word2id={}
        for each in self.id2word: #each 是self.id2word 字典的key 不是(key，value)组合
            self.word2id.setdefault(self.id2word[each],each)

    def load_train_data(self):

        with open(self.train_data_path,encoding='utf8') as fp:
            train_data_rawlines=fp.readlines()
        with open(self.train_label_path,encoding='utf8') as fp:
            train_label_rawlines=fp.readlines()
        total_lines = len(train_data_rawlines)
        assert len(train_data_rawlines)==len(train_label_rawlines)

        for index in range(total_lines):
            data_line = train_data_rawlines[index].split(" ")[:-1]
            label_line = train_label_rawlines[index].split(" ")[:-1]
            #assert len(data_line)==len(label_line)
            #align
            if len(data_line) < len(label_line):
                label_line=label_line[:len(data_line)]
            elif len(data_line)>len(label_line):
                data_line=data_line[:len(label_line)]
            assert len(data_line)==len(label_line)
            #add and seperate valid ,train set.
            data=[int(self.word2id.get(each,0)) for each in data_line]
            label=[int(self.state.get(each,self.state['O'])) for each in label_line]
            if random.uniform(0,1) <self.seperate_rate:
                self.valid_data_raw.append(data)
                self.valid_label_raw.append(label)
            else:
                self.train_data_raw.append(data)
                self.train_label_raw.append(label)
        self.train_batches= [i for i in range(int(len(self.train_data_raw)/self.batch_size) -1)]
        self.train_batch_index =0
        self.valid_batches=[i for i in range(int(len(self.valid_data_raw)/self.batch_size) -1) ]
        self.valid_batch_index = 0

    def load_test_data(self):

        with open(self.test_data_path,encoding='utf8') as fp:
            test_data_rawlines=fp.readlines()
        with open(self.test_label_path,encoding='utf8') as fp:
            test_label_rawlines=fp.readlines()
        total_lines = len(test_data_rawlines)
        assert len(test_data_rawlines)==len(test_label_rawlines)

        for index in range(total_lines):
            data_line = test_data_rawlines[index].split(" ")[:-1]
            label_line = test_label_rawlines[index].split(" ")[:-1]
            #assert len(data_line)==len(label_line)
            #align
            if len(data_line) < len(label_line):
                label_line=label_line[:len(data_line)]
            elif len(data_line)>len(label_line):
                data_line=data_line[:len(label_line)]
            assert len(data_line)==len(label_line)
            data=[int(self.word2id.get(each,0)) for each in data_line]
            label=[int(self.state.get(each,self.state['O'])) for each in label_line]
            self.test_data_raw.append(data)
            self.test_label_raw.append(label)
            
        self.test_batches= [i for i in range(int(len(self.test_data_raw)/self.batch_size) -1)]
        self.test_batch_index =0

    def handleInputData(self,text):
        input_data_raw=[]
        output_x = []
        words_x = []
        efficient_sequence_length=[]
        data_line = list(jieba.cut(text.strip()))
        sumCount = len(data_line)//self.sentence_length
        if len(data_line) % self.sentence_length : sumCount+=1
        for count in range(sumCount):
            input_data_raw.append(data_line[count*self.sentence_length:(count+1)*self.sentence_length])
        for idx in range(len(input_data_raw)):
            _data_line = input_data_raw[idx]
            _words_x = [word for word in _data_line]
            efficient_sequence_length.append(min(self.sentence_length,len(_data_line)))
            _data_trans = [int(self.word2id.get(each, 0)) for each in _data_line]
            #填充
            data = self.pad_sequence(_data_trans, self.sentence_length, 0)
            output_x.append(data)
            words_x.append(_words_x)
        return words_x,output_x,efficient_sequence_length

    def pad_sequence(self,sequence,object_length,pad_value=None):
        '''
        :param sequence: 待填充的序列
        :param object_length:  填充的目标长度
        :return:
        '''
        seq =copy.deepcopy(sequence[:object_length]) #若sequence过长就截断，
                                                          #若短于object_length就复制全部元素
        if pad_value is None:
            seq = seq*(1+int((0.5+object_length)/(len(seq))))
            seq = seq[:object_length]
        else:
            seq = seq+[pad_value]*(object_length- len(seq))
        return seq

    def next_train_batch(self):
        #padding
        output_x=[]
        output_label=[]
        efficient_sequence_length=[]
        index =self.train_batches[self.train_batch_index]
        self.train_batch_index =(self.train_batch_index +1 ) % len(self.train_batches)
        datas = self.train_data_raw[index*self.batch_size:(index+1)*self.batch_size]
        labels = self.train_label_raw[index*self.batch_size:(index+1)*self.batch_size]
        for index in range(self.batch_size):
            #复制填充
            data= self.pad_sequence(datas[index],self.sentence_length,0)
            label = self.pad_sequence(labels[index],self.sentence_length,0)
            #print("data len:%d"%(len(data)))
            #print("label len:%d"%(len(label)))
            output_x.append(data)
            output_label.append(label)
            efficient_sequence_length.append(min(self.sentence_length,len(labels[index])))
        return output_x,output_label,efficient_sequence_length
        #返回的都是下标,注意efficient_sequence_length是有效的长度

    def next_test_batch(self):
        #padding
        output_x=[]
        output_label=[]
        efficient_sequence_length=[]
        index =self.test_batches[self.test_batch_index]
        self.test_batch_index =(self.test_batch_index +1 ) % len(self.test_batches)
        datas = self.test_data_raw[index*self.batch_size:(index+1)*self.batch_size]
        labels = self.test_label_raw[index*self.batch_size:(index+1)*self.batch_size]
        for index in range(self.batch_size):
            #复制填充
            data= self.pad_sequence(datas[index],self.sentence_length,0)
            label = self.pad_sequence(labels[index],self.sentence_length,0)
            output_x.append(data)
            output_label.append(label)
            efficient_sequence_length.append(min(self.sentence_length,len(labels[index])))
        return output_x,output_label,efficient_sequence_length
        #返回的都是下标,注意efficient_sequence_length是有效的长度
 

    def next_valid_batch(self):
        output_x=[]
        output_label=[]
        efficient_sequence_length=[]
        index =self.valid_batches[self.valid_batch_index]
        self.valid_batch_index =(self.valid_batch_index +1 ) % len(self.valid_batches)
        datas = self.valid_data_raw[index*self.batch_size:(index+1)*self.batch_size]
        labels = self.valid_label_raw[index*self.batch_size:(index+1)*self.batch_size]
        for index in range(self.batch_size):
            #复制填充
            data= self.pad_sequence(datas[index],self.sentence_length,0)
            label = self.pad_sequence(labels[index],self.sentence_length,0)
            output_x.append(data)
            output_label.append(label)
            efficient_sequence_length.append(min(self.sentence_length,len(labels[index])))
        return output_x,output_label,efficient_sequence_length

    def count_entity(self,labels,lens):
    #输入是一个句子的标签
        start = -1#一个实体的起始位置
        size =0  #实体的长度
        rst = set()
        for i in range(lens):
            _state = self.id2state[labels[i]]
            if _state[0]=='B' or _state[0]=='S':
                start = i
                size =1
            elif start>=0:size+=1
            if _state[0]=='E' or _state[0]=='S':
                rst.add((labels[start],start,size))
                start=-1
                size=0
        if start>=0:
            rst.add((labels[start],start,size))
        return rst


    def evaluate(self,predict_labels,real_labels,efficient_length):
    #输入的单位是batch;
    # predict_labels:[batch_size,sequence_length],real_labels:[batch_size,sequence_length]
        sentence_nums =len(predict_labels) #句子的个数
        predict_cnt=0
        predict_right_cnt=0
        real_cnt=0
        for sentence_index in range(sentence_nums):
            predict_set=self.count_entity(predict_labels[sentence_index],efficient_length[sentence_index])
            real_set=self.count_entity(real_labels[sentence_index],efficient_length[sentence_index])
            right_=predict_set.intersection(real_set)
            predict_right_cnt+=len(right_)
            predict_cnt += len(predict_set)
            real_cnt +=len(real_set)
        return predict_right_cnt,predict_cnt,real_cnt

if __name__ == '__main__':
    dataGen = DATAPROCESS(train_data_path="data/source_data.txt",
                          train_label_path="data/source_label.txt",
                          test_data_path="data/test_data.txt",
                          test_label_path="data/test_label.txt",
                          word_embedings_path="data/source_data.txt.ebd.npy",
                          vocb_path="data/source_data.txt.vab",
                          batch_size=90,
                          seperate_rate=0.3
                        )
    datas,labels,efficient_sequence_length = dataGen.test_data()
    print(evaluate(labels,labels,efficient_sequence_length))
