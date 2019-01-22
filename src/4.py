#coding=utf-8
'''
Created on 2019年1月16日
https://blog.csdn.net/zhuzuwei/article/details/80777623
鬼吹灯文本挖掘4：LDA模型提取文档主题 sklearn LatentDirichletAllocation和gensim LdaModel
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





# 鬼吹灯文本挖掘4：LDA模型提取文档主题 sklearn LatentDirichletAllocation和gensim LdaModel
# 注：tfidf_mat数据准备可参考鬼吹灯文本挖掘3
import pickle
tfidf_mat=pickle.load(open('tfidf_mat.txt', 'rb'))

# 1. Sklearn实现LDA模型，并提取文档主题
#      （1）其中参数n_topics是主题个数，max_iter是迭代次数
#     （2）lda_model.components_中每行代表一个主题，每行中的每个元素代表对应词属于这个主题的得分
from sklearn.decomposition import LatentDirichletAllocation
n_topics = 8      # 自定义主题个数#DeprecationWarning: n_topics has been renamed to n_components in version 0.19 and will be removed in 0.21
lda_model = LatentDirichletAllocation(n_topics = n_topics, max_iter = 10)
# 使用TF-IDF矩阵拟合LDA模型
lda_model.fit(tfidf_mat)
 
# 拟合后模型的实质
print(lda_model.components_.shape)
print(lda_model.components_[:2])
 
# (8, 1654)
# Out[105]:
# array([[0.30237038, 0.29720752, 0.31504618, ..., 0.33985295, 0.2906448 ,
#         0.3043558 ],
#        [0.29870912, 0.30435234, 0.31793515, ..., 0.3215601 , 0.32073196,
#         0.31859002]])
# （3）其中argsort() 取元素的索引值，并将指最小的元素对应的索引值放在最前面，依次按元素值的大小顺序排列。
#         对列表操作[:-n_top_words-1:-1] 表示取最后n_top_words个元素后，再将这些元素顺序逆转。
#         即取元素值最大的前n_top_words个元素对应索引值。
# 主题词打印函数
def print_top_words(model, feature_names, n_top_words):
    for topic_idx,topic in enumerate(model.components_):
        print("Topic #%d:"%topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]))
        
    print()

c_vectorizer=pickle.load(open('c_vectorizer.txt', 'rb'))#我加的，来自前面
n_top_words = 12
tf_feature_names = c_vectorizer.get_feature_names()
print_top_words(lda_model, tf_feature_names, n_top_words)
#  
# Topic #0:
# 强烈 遭到 肚子 亲自 有名 打着 不可思议 全是 半截 遇到 一层 下去
# Topic #1:
# 将近 考古 详细 突然 大个子 人们 照明 听说 空气 说起 小孩 最后
# Topic #2:
# 故事 胖子 方向 沙漠 岔开 马上 也许 地下 沙子 不停 好像 激动
# Topic #3:
# 全都 接近 一天 研究 只见 接触 感觉 解放军 扔进 所有人 地下 漆黑
# Topic #4:
# 一个 我们 火焰 要塞 这座 棺材 看来 一边 上去 胖子 昏迷不醒 双手
# Topic #5:
# 时候 那么 超过 一面 里面 山石 胖子 两具 玉石 接近 什么 整整
# Topic #6:
# 外边 精绝 找个 最高 坚固 下边 遇到 进行 手上 沙漠 规模 一丝
# Topic #7:
# 我们 胖子 什么 一个 没有 沙漠 他们 就是 咱们 陈教授 但是 这些
# 
# 2. gensim实现LDA模型
#     （1）计算TF-IDF矩阵：
# 文档处理，提取主题词
import jieba, re
def my_cut2(inTxt):
    inTxt = re.sub('[a-zA-Z0-9]','',inTxt)
    jieba.lcut(inTxt)
    words_list = jieba.lcut(inTxt)
    return [w for w in words_list if w not in stop_words_dict and len(w) > 1]
    
gcd1_chap=pickle.load(open('gcd1_chap.txt', 'rb'))
chaplist = [my_cut2(w) for w in gcd1_chap.words_list]   # 形式为list of list

#15分钟入门NLP神器—Gensim
from gensim import corpora, models
 
dictionary = corpora.Dictionary(chaplist)
corpus = [dictionary.doc2bow(text) for text in chaplist]   # 仍为list of list
corpus    # 稀疏矩阵，第一个元素为词的序号，第二个为词频
 
tfidf_model = models.TfidfModel(corpus)  # 建立TF-IDF模型
corpus_tfidf = tfidf_model[corpus]        # 对所需文档计算TF-IDF结果, 不能直接被sklearn使用
print(corpus_tfidf)
# <gensim.interfaces.TransformedCorpus at 0x2c2941e7048>

#     （2）建立LDA模型：LdaModel，print_topics()模型打印出的主题不一定是按原文档的顺序打印的，而是按主题的重要程度              打印的。
from gensim.models import LdaModel
 
#列出所消耗时间备查
#%time是ipython的特殊功能，用于测试语句运行的时间。pip install ipython
# %time ldamodel = LdaModel(corpus_tfidf, id2word = dictionary, num_topics = 8, passes = 10)  # passes表示迭代循环次数
ldamodel = LdaModel(corpus_tfidf, id2word = dictionary, num_topics = 8, passes = 10)  # passes表示迭代循环次数
 
# 列出最重要的前若干个主题
a=ldamodel.print_topics()
print(a)
# [(0,
#   '0.001*"石匣" + 0.000*"先知" + 0.000*"羊皮" + 0.000*"陈教授" + 0.000*"预言" + 0.000*"沙漠" + 0.000*"石梁" + 0.000*"红犼" + 0.000*"尸香魔芋" + 0.000*"狼牙棒"'),
#  (1,
#   '0.000*"蝙蝠" + 0.000*"草原大地獭" + 0.000*"沙漠" + 0.000*"英子" + 0.000*"骆驼" + 0.000*"冲锋枪" + 0.000*"安力满" + 0.000*"要塞" + 0.000*"黑沙漠" + 0.000*"棺材"'),
#  (2,
#   '0.000*"胡国华" + 0.000*"孙先生" + 0.000*"野人" + 0.000*"燕子" + 0.000*"郝爱国" + 0.000*"屯子" + 0.000*"山口" + 0.000*"城中" + 0.000*"安力满" + 0.000*"插队"'),
#  (3,
#   '0.000*"幻觉" + 0.000*"耳光" + 0.000*"摸金符" + 0.000*"制造" + 0.000*"解开" + 0.000*"盗墓者" + 0.000*"大祭司" + 0.000*"搞清楚" + 0.000*"圣者" + 0.000*"药物"'),
#  (4,
#   '0.000*"洛宁" + 0.000*"大个子" + 0.000*"班长" + 0.000*"指导员" + 0.000*"火球" + 0.000*"雪崩" + 0.000*"赵萍萍" + 0.000*"霸王蝾螈" + 0.000*"平台" + 0.000*"石柱"'),
#  (5,
#   '0.000*"野猪" + 0.000*"洛宁" + 0.000*"九层妖楼" + 0.000*"刘工" + 0.000*"云母" + 0.000*"殉葬" + 0.000*"先知" + 0.000*"骆驼" + 0.000*"沙漠" + 0.000*"先圣"'),
#  (6,
#   '0.000*"胡国华" + 0.000*"野猪" + 0.000*"老鼠" + 0.000*"英子" + 0.000*"舅舅" + 0.000*"石像" + 0.000*"野人" + 0.000*"王二" + 0.000*"杠子" + 0.000*"大金牙"'),
#  (7,
#   '0.000*"大金牙" + 0.000*"洛阳铲" + 0.000*"摸金校尉" + 0.000*"倒斗" + 0.000*"奉献" + 0.000*"唇典" + 0.000*"匣子" + 0.000*"元良" + 0.000*"黑道" + 0.000*"铁钎"')]


# （3） 检索和文本内容最接近的主题
#       a. 根据词频向量进行计算
# 检索和文本内容最接近的主题
query = gcd1_chap.txt[1]       # 检索和第一回最接近的主题
query_bow = dictionary.doc2bow(my_cut2(query))       # 频数向量
query_idf = tfidf_model[query_bow]                 # TF-IDF向量
 
a=ldamodel.get_document_topics(query_bow)            # 需要输入和文档对应的bow向量
print(a)
# [(0, 0.030599635), (6, 0.96860933)]
a=ldamodel[query_bow]         # 和以下语句的结果一致 ldamodel.get_document_topics(query_bow)
print(a) 
# [(0, 0.030602157), (6, 0.9686068)]
#      b. 根据TF-IDF向量进行计算 
a=ldamodel.get_document_topics(query_idf) 
print(a)
# [(6, 0.93530923)]
 
a=ldamodel[query_idf]            # 和以下语句的结果一致 ldamodel.get_document_topics[query_idf]
print(a)
# [(6, 0.9353105)]
#     c. 验证：可以看出 "字符串query和第6个主题高度接近" 这个结论还是比较靠谱的
