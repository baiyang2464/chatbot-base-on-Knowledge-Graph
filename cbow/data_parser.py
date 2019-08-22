
# coding: utf-8
import  random
class TextLoader(object):
    def __init__(self,input_data_path,Context_length=10,batch_size=10,min_frq = 2):
        self.Context_length = Context_length #定义上下文的长度
        self.V =  {} # map word to index
        self.inverseV ={} #map index to word
        self.raw_text =list()
        self.x_data =list()
        self.y_data =list()
        self.number_batch = 0
        self.batch_size = batch_size

        raw_text = []
        #输入原始数据并统计词频
        V = dict() #{'word':frq}
        print("loading data...")
        linecount=1
        with open(input_data_path,"r",encoding="utf8") as fp:
            lines = fp.readlines()
            for line in lines:
                line =line.split(" ")
                line = ['<START>']+line+['<END>'] #为每句话加上<START>,<END>
                raw_text += line
                for word in line:
                    if word in V:
                        V[word] +=1
                    else:
                        V.setdefault(word,1)
                if linecount % 50000==0:
                    print("加载%d行数据"%(linecount))
                linecount+=1
                
        
        print("建立词到id的字典...")
        #清除词频太小的词,同时为各个词建立下标到索引之间的映射
        self.V.setdefault('<UNK>',0)
        self.inverseV.setdefault(0,'<UNK>')
        cnt = 1
        for word in V:
            if V[word] <= min_frq:
                continue
            else:
                self.V.setdefault(word,cnt)
                self.inverseV.setdefault(cnt,word)
                cnt +=1
        self.vacb_size = len(self.V)
        #将文本由字符串,转换为词的下标
        for word in raw_text:
            self.raw_text +=[self.V[word] if word in self.V else self.V['<UNK>']]
        #生成batches
        self.gen_batch()

    def gen_batch(self):
        self.x_data =[]
        self.y_data =[]
        print("产生批数据...")
        for index in range(self.Context_length,len(self.raw_text)-self.Context_length):
            #index的前Context,加上index的后Context个词,一起构成了index词的上下文
            x = self.raw_text[(index-self.Context_length):index]+ self.raw_text[(index+1):(self.Context_length+index)]
            y = [ self.raw_text[index] ]
            self.x_data.append(x)
            self.y_data.append(y)
        self.number_batch =int( len(self.x_data) / self.batch_size)

    def next_batch(self):
        batch_pointer =  random.randint(0,self.number_batch-1)
        x = self.x_data[batch_pointer:(batch_pointer+self.batch_size)]
        y = self.y_data[batch_pointer:(batch_pointer+self.batch_size)]
        return x ,y

if __name__ == '__main__':
    data = TextLoader("./cbow/document.txt")
    print(data.vacb_size)
    x,y=data.next_batch()
    print(x)
    print(y)

