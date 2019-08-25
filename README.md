It is indeed a luxury to keep human reason forever. by  Moss, a robot of the film The Wandering Earth

“让人类永远保持理智，确实是一种奢求” ，机器人莫斯，《流浪地球》

<p align="center">
	<img src=./pictures/moss.jpg alt="Sample"  width="600">
	<p align="center">
		<em> </em>
	</p>
</p>
###  项目概况

本项目为一个使用深度学习方法解析问题，知识图谱存储、查询知识点，基于医疗垂直领域的对话系统的后台程序

+ 运行效果：



+ 项目的搭建大致分为四个模块：
  + 基础数据爬取
  + 知识图谱构建
  + 问句分析
  + 回答生成

+ 项目运行环境：

python   :  python 3.6.8

运行系统：ubuntu 16.04 

知识图谱：neo4j 3.2.2 图形数据库

```
neo4j对应的python驱动
py2neo          3.1.1  
```

深度学习：

```
jieba           0.39   
numpy           1.17.0 
pandas          0.25.0 
tensorflow      1.10.0 
```

文本匹配：ahocorasick （安装方法 pip install pyahocorasick）

必要说明：

深度学习模块深度网络的训练使用tensorflow的gpu版本，在应用阶段由于要部署要服务器上使用的对应的tensorflow的cpu版本

若要clone项目，尽量保持扩展包的版本一致

+ 项目主要文件目录结构

```shell
chatbot
├── answer_search.py                        # 问题查询及返回
├── BiLSTM_CRF.py                           # 实体识别的双向LSTM-CRF网络
├── build_medicalgraph.py                   # 将结构化json数据导入neo4j
├── chatbot_graph.py                        # 问答程序脚本
├── classifyApp.py                          # 问句分类应用脚本
├── classifyUtils.py                        # 工具函数集合
├── data
│   └── medical.json                        # 全科知识数据
├── data_ai
│   ├── cbowData                            # 词向量文件
│   │   ├── classifyDocument.txt.ebd.npy    # 词向量查找表
│   │   ├── classifyDocument.txt.vab        # 词向量中词与索引对照表
│   │   ├── document.txt.ebd.npy            
│   │   └── document.txt.vab
│   ├── classifyData                        # 问句分类训练数据
│   │   ├── test_data.txt
│   │   └── train_data.txt
│   ├── classifyModel                       # 问句分类模型
│   │   ├── checkpoint
│   │   ├── model-3500.data-00000-of-00001
│   │   ├── model-3500.index
│   │   └── model-3500.meta
│   ├── nerData                          
│   └── nerModel                            # 命名实体识别模型
├── dict                                    # 实体数据文件
├── nerApp.py                               # 命名实体识别应用脚本
├── nerUtils.py                             # 工具函数集合
├── prepare_data                           
│   ├── build_data.py                       # 数据库操作脚本
│   ├── data_spider.py                      # 数据采集脚本
│   └── max_cut.py                          # 基于词典的最大前向/后向匹配
├── question_analysis.py                    # 问句类型分类脚本
├── question_parser.py                      # 回答生成脚本
└── text_cnn.py                             # 文本分类的cnn网络
```



### 基础数据爬取&知识图谱搭建

基础数据爬取于[寻医问药](<http://www.xywy.com/>)网站，一家医疗信息提供平台，上面的数据做了较好的分类处理，爬下来后可以较为方便的保存为json格式的结构化文件，格式展示如下：

<p align="center">
	<img src=./pictures/json_show.gif alt="Sample"  width="700">
	<p align="center">
		<em> 爬取的数据保存为json格式文件 </em>
	</p>
</p>








​	