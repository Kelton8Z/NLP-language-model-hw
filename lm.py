#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Zimeng Qiu <zimengq@andrew.cmu.edu>

"""
F19 11-411/611 NLP Assignment 3 Task 1
N-gram Language Model Implementation Script
Kinjal Jain Feb 2020

This is a simple implementation of N-gram language model

Write your own implementation in this file!
"""

import argparse

from utils import *
from collections import Counter
import copy
import math

class LanguageModel(object):
    """
    Base class for all language models
    """
    def __init__(self, corpus, ngram, min_freq, uniform=False,dic={},one={}):
        """
        Initialize language model
        :param corpus: input text corpus to build LM on
        :param ngram: number of n-gram, e.g. 1, 2, 3, ...
        :param min_freq: minimum frequency threshold to set a word to UNK placeholder
                         set to 1 to not use this threshold
        :param uniform: boolean flag, set to True to indicate this model is a simple uniform LM
                        otherwise will be an N-gram model
        """
        self.corpus = corpus
        self.ngram = ngram
        self.min_freq = min_freq
        self.uniform = uniform
        self.dic = self.build()[1]
        self.one = self.build()[0]

    def build(self):
        """
        Build LM from text corpus
        """
        one = {}
        for sentence in self.corpus:
            for word in sentence:
                if word in one:
                    one[word] += 1
                else:
                    one[word] = 1

        for key,val in one.copy().items():
            one[key] += 1
            if val < self.min_freq:
                one['UNK'] = one.pop(key)



        if self.ngram == 2:
            biDictFromTrain = {}
            corpusLen = len(self.corpus)
            for i in range(corpusLen):
                sentenceLen = len(self.corpus[i])
                prevWord = self.corpus[i][0]
                for j in range(1,sentenceLen):
                    word = self.corpus[i][j]
                    cor = prevWord + " " + word
                    prevWord = word
                    #c = Counter(cor)
                    if cor in biDictFromTrain:
                        biDictFromTrain[cor] += 1
                    else:
                        biDictFromTrain[cor] = 1
            return [one,biDictFromTrain]

        if self.ngram == 1:
            uniDictFromTrain = {}
            for sentence in self.corpus:
                for word in sentence:
                    if word in uniDictFromTrain:
                        uniDictFromTrain[word] += 1
                    else:
                        uniDictFromTrain[word] = 1
            return [one,uniDictFromTrain]

        if self.ngram == 3:
            triDictFromTrain = {}
            corpusLen = len(self.corpus)
            for i in range(corpusLen):
                sentenceLen = len(self.corpus[i])
                firstWord = self.corpus[i][0]
                for j in range(1,sentenceLen-1):
                    secondWord = self.corpus[i][j]
                    cor = firstWord + " " + secondWord
                    firstWord = secondWord
                    k = j+1
                    thirdWord = self.corpus[i][k]
                    cor += " " + thirdWord
                    secondWord = thirdWord
                    if cor in triDictFromTrain:
                        triDictFromTrain[cor] += 1
                    else:
                        triDictFromTrain[cor] = 1

            return [one,triDictFromTrain]

    def most_common_words(self, k):
        """
        This function will only be called after the language model has been built
        Your return should be sorted in descending order of frequency
        Sort according to ascending alphabet order when multiple words have same frequency
        :return: list[tuple(token, freq)] of top k most common tokens
        """
        wordCount = Counter(self.dic)
        return wordCount.most_common(k)

def calculate_perplexity(models, coefs, data):

    """
    Calculate perplexity with given model
    :param models: language models
    :param coefs: coefficients
    :param data: test data
    :return: perplexity
    """

    perplex = 1
    unigram = models[1]
    bigram = models[2]
    textSize = len(data)
    for biword,freq in bigram.dic.items():
        #print("bigram")
        #print(biword)
        #print("\n")
        firstWord = biword.split(" ")[0]
        #print("1st word")
        #print(firstWord)
        #print("\n")
        if firstWord in unigram.one:
            #print("biwordFreq")
            #print(bigram.dic[biword])
            #print("\n")
            #print("unigramFreq")
            #print(unigram.one[firstWord])
            #print("\n")
            curProb = math.log(bigram.dic[biword]/(unigram.one[firstWord]+2),2)
        else:
            firstWord = "UNK"
            curProb = math.log(bigram.dic[biword]/(unigram.one[firstWord]+2),2)
        #print("curProb")
        #print(curProb)
        perplex += curProb
        #print(biword)
        #print("\n")
        #print("prevWord")
        #print(biword.split(" ")[0])
        #print("\n")
    #print("\n")
    #print("perplex")
    #print(perplex)
    #print("\n")
    perplex = 2**(-1/textSize * perplex)

    return perplex

# Do not modify this function!
def parse_args():
    """
    Parse input positional arguments from command line
    :return: args - parsed arguments
    """
    parser = argparse.ArgumentParser('N-gram Language Model')
    parser.add_argument('coef_unif', help='coefficient for the uniform model.', type=float)
    parser.add_argument('coef_uni', help='coefficient for the unigram model.', type=float)
    parser.add_argument('coef_bi', help='coefficient for the bigram model.', type=float)
    parser.add_argument('coef_tri', help='coefficient for the trigram model.', type=float)
    parser.add_argument('min_freq', type=int,
                        help='minimum frequency threshold for substitute '
                             'with UNK token, set to 1 for not use this threshold')
    parser.add_argument('testfile', help='test text file.')
    parser.add_argument('trainfile', help='training text file.', nargs='+')
    args = parser.parse_args()
    return args


# Main executable script provided for your convenience
# Not executed on autograder, so do what you want
if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # load and preprocess train and test data
    train = preprocess(load_dataset(args.trainfile))
    test = preprocess(read_file(args.testfile))

    # build language models
    uniform = LanguageModel(train, ngram=1, min_freq=args.min_freq, uniform=True)
    unigram = LanguageModel(train, ngram=1, min_freq=args.min_freq)
    bigram = LanguageModel(train, ngram=2, min_freq=args.min_freq)
    trigram = LanguageModel(train, ngram=3, min_freq=args.min_freq)

    # calculate perplexity on test file
    ppl = calculate_perplexity(
        models=[uniform, unigram, bigram, trigram],
        coefs=[args.coef_unif, args.coef_uni, args.coef_bi, args.coef_tri],
        data=test)

    print("Perplexity: {}".format(ppl))






