#coding=utf-8
'''
Created on 2019年1月16日
https://blog.csdn.net/zhuzuwei/article/details/80766563
鬼吹灯文本挖掘2：wordcloud 词云展示
@author: admin
'''
# 1.  准备数据：具体可参考前一篇分析：鬼吹灯文本挖掘1
import wordcloud #pip install wordcloud
import matplotlib.pyplot as plt
myfont = r'C:\Windows\Fonts\simkai.ttf'   # 获取本地已安装字体
 
import pickle
gcd1_words_list=pickle.load(open('gcd1_words_list.txt', 'rb'))

from imageio import imwrite,imread
my_mask = imread('chineseMap.jpg')
cloudobj = wordcloud.WordCloud(font_path=myfont, mask=my_mask, max_words=100,\
                               mode='RGBA', background_color=None).generate(gcd1_words_list[15])
imgobj = imread('IMG_9475.png') #.jpg
image_colors = wordcloud.ImageColorGenerator(imgobj)      # 获取图片颜色
cloudobj.recolor(color_func=image_colors)     # 重置词云颜色
 
plt.imshow(cloudobj)
plt.axis('off')
plt.show()
plt.close()
