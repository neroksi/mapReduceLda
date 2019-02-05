#!/usr/bin/env python
# coding: utf-8


from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import re
from unidecode import unidecode


stpwd = set(stopwords.words("english"))
stemmer = SnowballStemmer("english").stem



def alphaAndSpaceOnly(s):
    return re.sub(r'[\d]+', '', re.sub(r'[^\w\s]+', '', s))

def removeMultipleSpaces(s):
    return re.sub(r"\s\s+", " ", s) 


def toAscii(s):
    return unidecode(s)

def getTokens(s):
    return s.split()

def removeStopwords(tokens):
    return set(w for w in tokens if w not in stpwd and len(w) > 1 )


def getStem(tokens):
    return list(set(stemmer(w)  for w in tokens))

def preprocessAndGetTokens(doc):
    doc = doc.lower()
    doc = toAscii(doc)
    doc = alphaAndSpaceOnly(doc)
    doc = removeMultipleSpaces(doc)
    tokens = getTokens(doc)
    tokens = removeStopwords(tokens)
    tokens = getStem(tokens)
    return tokens