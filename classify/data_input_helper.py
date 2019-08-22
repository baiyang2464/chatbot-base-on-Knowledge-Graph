import numpy as np
import re
import  json

class data_process:
    def __init__(self,train_data_path,word_embedings_path,vocb_path,num_classes,max_document_length,dev_sample_percentage=0.2):
        self.train_data_path =train_data_path
        self.word_embedding_path = word_embedings_path
        self.vocb_path  = vocb_path
        self.num_classes = num_classes
        self.max_document_length = max_document_length

        self.word_embeddings=None
        self.id2word={}
        self.word2id={}
        self.embedding_length =0
        self.dev_sample_percentage = dev_sample_percentage

    def load_wordebedding(self):
        self.word_embeddings = np.load(self.word_embedding_path)
        self.embedding_length = np.shape(self.word_embeddings)[-1]
        with open(self.vocb_path, encoding="utf8") as fp:
            self.id2word = json.load(fp)
        self.word2id = {}
        for each in self.id2word:  # each 是self.id2word 字典的key 不是(key，value)组合
            self.word2id.setdefault(self.id2word[each], each)

    def load_raw_data(self, filepath):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        # Load data from files
        train_datas = []
        with open(filepath, 'r', encoding='utf-8',errors='ignore') as f:
            train_datas = f.readlines()
        one_hot_labels = []
        x_datas = []
        for line in train_datas:
            parts = line.encode('utf-8').decode('utf-8-sig').strip().split(' ',1)
            if len(parts)<2 or (len(parts[1].strip()) == 0):
                continue
            x_datas.append(parts[1])
            one_hot_label = [0]*self.num_classes
            label = int(parts[0])
            one_hot_label[label] = 1
            one_hot_labels.append(one_hot_label)
        print (' data size = ' ,len(train_datas))
        return [x_datas, np.array(one_hot_labels)]

    def load_data(self):
        """Loads starter word-vectors and train/dev/test data."""
        print("Loading word2vec and textdata...")
        x_text, y = self.load_raw_data(self.train_data_path)

        max_document_length = max([len(x.split(" ")) for x in x_text])
        print('len(x) = ', len(x_text), ' ', len(y))
        print(' max_document_length = ', max_document_length)
        x = []
        x = self.get_data_idx(x_text)
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        dev_sample_index = -1 * int(self.dev_sample_percentage * float(len(y)))
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

        return x_train, x_dev, y_train, y_dev

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)

                #print('epoch = %d,batch_num = %d,start = %d,end_idx = %d' % (epoch,batch_num,start_index,end_index))
                yield shuffled_data[start_index:end_index]

    def get_data_idx(self,text):
        """
        Gets index of input data to generate word vector.
        """
        text_array = np.zeros([len(text), self.max_document_length], dtype=np.int32)
        total_lines = len(text)
        for index in range(total_lines):
            data_line = text[index].split(" ")[:-1]
            for pos in range(min(len(data_line),self.max_document_length)):
                text_array[index,pos] = int(self.word2id.get(data_line[pos],0))
        return text_array
    
    def handle_input(self,text):
        text_array = np.zeros([1, self.max_document_length], dtype=np.int32)
        data_line= text.strip().split(" ")
        for pos in range(min(len(data_line),self.max_document_length)):
            text_array[0, pos] = int(self.word2id.get(data_line[pos], 0))
        return text_array

    def evalution(self,confusion_matrix):
        """
        Gets evalution:precission,recall and f1_score
        """
        # tensorflow confusion_matrix api:https://haosdent.gitbooks.io/tensorflow-document/content/api_docs/python/contrib.metrics.html#confusion_matrix.
        # 所计算出来的混淆矩阵，列是真实值（也就是期望值），行是预测值
        accu = [0]*self.num_classes
        column = [0]*self.num_classes
        line = [0]*self.num_classes
        recall = 0
        precision = 0
        for i in range(0,self.num_classes):
            accu[i] = confusion_matrix[i][i]
        for i in range(0,self.num_classes):
            for j in range(0,self.num_classes):
                column[i]+=confusion_matrix[j][i]
        for i in range(0,self.num_classes):
            for j in range(0,self.num_classes):
                line[i]+=confusion_matrix[i][j]
        for i in range(0,self.num_classes):
            if column[i] != 0:
                recall+=float(accu[i])/column[i]
        recall = recall / self.num_classes
        for i in range(0,self.num_classes):
            if line[i] != 0:
                precision+=float(accu[i])/line[i]
        precision = precision / self.num_classes
        f1_score = (2 * (precision * recall)) / (precision + recall)
        return precision,recall,f1_score

if __name__ == "__main__":
    x_text, y = load_data_and_labels('')
    print (len(x_text))