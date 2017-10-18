#! /usr/bin/env python
# coding:utf-8

import random


def make_score(num):
    return [random.randint(0, 100) for i in range(num)]


def less_average(scores):
    num = len(scores)
    ave_score = sum(scores) / float(num)
    less_ave = [score for score in scores if score < ave_score]
    return (ave_score, len(less_ave))


if __name__ == '__main__':
    score = make_score(40)
    average_num, less_num = less_average(score)
    print('the score of average is: {}'.format(average_num))
    print('the number of less average is: {}'.format(less_num))
    print('the every score is[from big to small]: {}'.format(sorted(score, reverse=True)))
