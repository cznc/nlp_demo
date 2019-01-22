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

class GroupedColorFunc(object):
    """Create a color function object which assigns DIFFERENT SHADES of
       specified colors to certain words based on the color to words mapping.

       Uses wordcloud.get_single_color_func

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)

    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)
