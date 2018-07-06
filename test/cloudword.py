#!/usr/bin/env python
# -*- coding: utf-8 -*-

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import jieba
from scipy.misc import imread


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def move_stopwords(sentence):
    stopwords = stopwordslist('./stopwords.txt')
    outstr = []
    for word in sentence:
        if word != '\t' and word != '\n' and word != '\n\n' and word not in stopwords:
            outstr.append(word)
    return outstr


text = open(u'text.txt', 'r').read()
jieba.suggest_freq("熊节", tune=True)
wordlist = jieba.cut(text, cut_all=True)

wl = move_stopwords(wordlist)
wds = ' '.join(wl)

img = imread('./redar.png')



font = r'./fonts/simkai.ttf'
wordcloud = WordCloud(background_color='white',
                      font_path=font,
                      mask=img,
                      # width=100,
                      # height=860,
                      margin=2,
                      max_words=500).generate(wds)

plt.imshow(wordcloud)
plt.axis("off")
plt.show()
wordcloud.to_file('agile.png')


