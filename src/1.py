#coding=utf-8
'''
Created on 2019年1月16日
https://blog.csdn.net/zhuzuwei/article/details/80766106
鬼吹灯文本挖掘1：jieba分词和CountVectorizer向量化
1. 数据预处理
@author: admin
'''
import pandas as pd
import numpy as np
import jieba
import re
noval_txt='gcd_jjgc.txt'

# 1. 读取文档: 返回dataframe对象。 DataFrame 来自R语言。pandas=DataFrame + Series 。
gcd1_df = pd.read_csv(noval_txt,sep='aaa',encoding='utf-8',names=['txt'], engine='python')
 
# 章节判断用变量预处理
def is_chap_head(tmpstr):
    pattern = re.compile('第.{1,7}章')
    length= len(pattern.findall(tmpstr))
#     print('章数='+str(length)+tmpstr[0:10])#第七章 大冰川
    return length
 
gcd1_df['is_chap_head'] = gcd1_df.txt.apply(is_chap_head)
# raw['chap']  = 0  #初始化所有章节为0
 
 # 章节判断
chap_num = 0
for i in range(len(gcd1_df)):
    if gcd1_df['is_chap_head'][i] == 1:
        chap_num += 1
    gcd1_df.loc[i,'chap'] = chap_num
 
del gcd1_df['is_chap_head']

print('章数='+str(chap_num))

###2. 分词
# 2.1. 获取停用词库
my_stop_words_path = '停用词.txt' #'stopword.txt'
stop_words_dict = []
with open(my_stop_words_path, errors='ignore') as fr:
    for line in fr.readlines():
        stop_words_dict.append(line.strip())
print('停用词数={}'.format(len(stop_words_dict)))
 
# 2.2. 加载搜狗中的鬼吹灯词库
gcd_words_dict_path = '词库.txt' #'鬼吹灯词库.txt'
jieba.load_userdict(gcd_words_dict_path)
# 2.3. 自定义分词函数
def my_cut(inTxt):
    inTxt = re.sub('[a-zA-Z0-9]','',inTxt)
    jieba.lcut(inTxt)
    words_list = jieba.lcut(inTxt)
    return ' '.join([w for w in words_list if w not in stop_words_dict and len(w) > 1])
#gcd1_df.txt.apply(my_cut) 是对gcd1_df.txt中的每条记录应用 my_cut 函数
import datetime as datatime
time1=datatime.datetime.now();
gcd1_df['words_list'] = gcd1_df.txt.apply(my_cut) #主要耗时在这里 0:00:06.343362
time2=datatime.datetime.now();
print('耗时={}'.format(time2-time1))
gcd1_df.head()

###3.  向量化：利用sklearn实现
# 3.1. 针对chap合并
gcd1_chap = gcd1_df.groupby(['chap']).sum()
gcd1_chap.head()# 数据变成了这样:
# txt    words_list                                                                              
# chap                                                                                        
# 1.0    第一章 白纸人和鼠友我的祖父叫胡国华，胡家祖上是十里八乡有名的大地主，最辉煌的时期在城里买了...
# 2.0    第二章 《十六字阴阳风水秘术》从那以后胡国华就当了兵，甚得重用，然而在那个时代，天下大乱，军...
# 3.0    第三章 大山里的古墓虽说是内蒙，其实离黑龙江不远，都快到外蒙边境了。居民也以汉族为主，只有少...
# 4.0    第四章 昆仑不冻泉那一年的春天，中国政府的高层因为感受到国际敌对势力的威胁，不断进行战略上的...
# 5.0    第五章 火瓢虫进山的第三天早晨，小分队抵达了大冰川，传说这附近有一个                         
import jieba.analyse #没有这行报错: module 'jieba' has no attribute 'analyse'
a=jieba.analyse.extract_tags(gcd1_chap.txt[1])
b=jieba.analyse.extract_tags(gcd1_chap.txt[3], withWeight = True, topK=10)   # 要求返回权重值
print(a)
print(b)
# ['胡国华', '舅舅', '纸人', '白纸', '媳妇', '外甥', '大烟', '女人', '门帘', '大洋', '十三里', '见客', '吃喝嫖赌', '不过', '但是', '贤惠', '福寿', '心里', '学好', '败家子']
# [('胡国华', 0.26770086241647695), ('孙先生', 0.10664678551083283), ('女尸', 0.0900293140844218), ('棺材', 0.05079270251431994), ('小翠', 0.050327047431036546), ('插队', 0.04072927523645896), ('十三里', 0.03446010299340922), ('心肝', 0.03261484930378669), ('荒坟', 0.032122899405632115), ('坟地', 0.028032902545566205)]

#3.2sklearn中的CountVectorizer可以实现将文本转换为稀疏矩阵，此处输入的中文文本必须是要先分好词再按空格分隔
#合并为一个字符串才可以。参数min_df=5表示词必须要在至少5个文档中出现过，否则就不考虑。
gcd1_words_list = list(gcd1_chap.words_list)
import pickle #我增加的
pickle.dump(gcd1_words_list,open('gcd1_words_list.txt', 'wb'))#我增加的
pickle.dump(gcd1_chap,open('gcd1_chap.txt', 'wb'))#我增加的， 用于第四篇

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(min_df=5)  #  词至少在5个文档中出现过
gcd1_words_vec = count_vect.fit_transform(gcd1_words_list)
# (3) 此稀疏矩阵可以方便地转换为数组/矩阵。 可用下面的代码查看：
gcd1_words_vec.toarray()
gcd1_words_vec.todense()
a=count_vect.get_feature_names()[-10:]
print('count_vect.get_feature_names()[-10:]:')
print(a)
#作者的是这样: ['黄沙', '黑乎乎', '黑暗', '黑沙漠', '黑漆漆', '黑色', '黑蛇', '黑风口', '黑驴蹄子', '鼻子']
#俺们的是这样: ['魔鬼', '鲜血', '麻烦', '黄沙', '黑暗', '黑漆漆', '黑色', '黑蛇', '黑驴', '鼻子']

if __name__ == '__main__':
    pass
