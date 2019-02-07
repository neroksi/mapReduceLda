#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from json import loads
from numpy import array

def processPaperAbstract(docstr):
    doc = loads(docstr)
    abstract = doc["paperAbstract"]
    tokens = array(preprocessAndGetTokens(abstract))
    return (doc["id"], tokens)

