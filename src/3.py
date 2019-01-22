#coding=utf-8
'''
Created on 2019年1月16日
https://blog.csdn.net/zhuzuwei/article/details/80775078
鬼吹灯文本挖掘3:
@author: admin
'''
# 1.  准备数据：具体可参考前一篇分析：鬼吹灯文本挖掘1
import wordcloud #pip install wordcloud
import matplotlib.pyplot as plt
myfont = r'C:\Windows\Fonts\simkai.ttf'   # 获取本地已安装字体
 
import pickle
gcd1_words_list=pickle.load(open('gcd1_words_list.txt', 'rb'))

# 2.1. 获取停用词库 #直接拷贝的
my_stop_words_path = '停用词.txt' #'stopword.txt'
stop_words_dict = []
with open(my_stop_words_path, errors='ignore') as fr:
    for line in fr.readlines():
        stop_words_dict.append(line.strip())
print('停用词数={}'.format(len(stop_words_dict)))

# 2. Sklearn计算词频矩阵
#      CountVectorizer可以将文本列表转换为词频矩阵sparse matrix，且为稀疏矩阵，其中参数min_df = 5 指定筛选出至少在5篇文档中出现过的词。       words_count_mat.todense() 可将稀疏矩阵转换为标准矩阵。
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer, TfidfVectorizer
c_vectorizer = CountVectorizer(min_df = 5)
words_count_mat = c_vectorizer.fit_transform(gcd1_words_list)      # 将文本列表转换为词频矩阵
print(words_count_mat)
a=words_count_mat.todense()
print(a)
# 3. Sklearn 计算 TF-IDF 矩阵
#    (1) 使用TfidfTransformer:  可以将词频矩阵转换为TF-IDF矩阵
tfidf_vectorizer = TfidfTransformer()
tfidf_mat = tfidf_vectorizer.fit_transform(words_count_mat)

import pickle#我增加的，用于第四篇
pickle.dump(tfidf_mat,open('tfidf_mat.txt', 'wb'))#我增加的

print(tfidf_mat)
a=tfidf_mat.todense()
print(a)
print('字典长度:{}'.format(len(c_vectorizer.vocabulary_)))
a=c_vectorizer.vocabulary_
print(a)
pickle.dump(c_vectorizer,open('c_vectorizer.txt', 'wb'))#我增加的， 用于第四篇
# （2）使用TfidfVectorizer: 可以将文本列表直接转换为TF-IDF矩阵，相当于CountVectorizer + TfidfTransformer的效果
tfidf_vectorizer = TfidfVectorizer(min_df=5) #   CountVectorizer + TfidfTransformer
tfidf_mat2 = tfidf_vectorizer.fit_transform(gcd1_words_list)
print(tfidf_mat2)
