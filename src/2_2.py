#coding=utf-8
'''
Created on 2019年1月16日
https://blog.csdn.net/zhuzuwei/article/details/80766563
鬼吹灯文本挖掘2：wordcloud 词云展示: 给关键字着色: 没通
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
 
 
cloudobj = wordcloud.WordCloud(font_path=myfont,width=1200,prefer_horizontal=0.9, height=800,\
                               mode='RGBA',background_color=None,stopwords=stop_words_dict, \
                               max_words=100).generate(' '.join(gcd1_words_list))

# 指定分组色系
color_to_words = {
                'blue':['插队','女尸','军阀','身上'],
                'red':['胖子','胡国华','孙先生','棺材']
}
default_color = 'green'     #其余单词的默认颜色

from srcnlp.wcutils import GroupedColorFunc as GroupedColorFunc
grouped_color_func = GroupedColorFunc(color_to_words,default_color)
 
cloudobj.recolor(color_func=grouped_color_func)
 
plt.imshow(cloudobj)
plt.axis('off')
plt.show()
plt.close()
