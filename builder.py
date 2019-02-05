#!/usr/bin/env python
# coding: utf-8

import numpy as np
from datetime import datetime
import pickle as pkl, json, os
from fileUtils import load, saveByPartition, dump

def initDocCounts(ind, part, nbDocs,nbTopics):
    """Initialise countDocs,the  document-topic matrix. One per partition"""
    countDocs = np.zeros((nbDocs, nbTopics))
    for el in part :
        count = np.bincount(el[2], minlength=nbTopics)
        countDocs[el[0]] = count
        
    file =  "matrix/countDocs/docs__%04d__"%ind
    with open(file, "wb") as f :
        pkl.dump(countDocs,f)
    return []



def initWordCounts(ind, part, nbVocabs,nbTopics):
    """Initialise countWords, the wors-topic matrix. One per partition"""
    
    countWords = np.zeros((nbVocabs, nbTopics))
    
    for el in part :
        countWords[el[1], el[2]] += 1
            
    file = "matrix/countWords/words__%04d__"%ind
    with open(file, "wb") as f :
        pkl.dump(countWords,f)
    return []



def initCountWordsAll(vocabs, nvocabAll, nbTopics):
    """Initialise countWords, the global word-topic matrix and update local countWords matrices"""
    
    countWords = np.zeros((nvocabAll, nbTopics))
    nbPartitions = len(vocabs)
    for ind in range(nbPartitions):
        countWords_ind = load("matrix/countWords/words__%04d__"%ind)
        table = np.array(list(vocabs[ind].values()))
        countWords[table[:,1]] += countWords_ind[table[:,0]]
    dump(countWords , "matrix/countWords/words_all")
    
    #Now let update each countWords to its correct state (the old values are wrong as they
    # don't take into account the docs in other partitions)
    for ind in range(nbPartitions):
        table = np.array(list(vocabs[ind].values()))
        dump(countWords[table[:,1]], "matrix/countWords/words__%04d__"%ind)



def updateCountWordsAll() :
    """Update the global countWords matrix"""
    vocab_files = sorted([ "matrix/vocabulary/" + file for file in os.listdir("matrix/vocabulary/") if "vocab__" in file])
    vocabs = [load(file) for file in vocab_files]
    nbPartitions  = len(vocabs)
    countWords = load( "matrix/countWords/words_all")

    for ind in range(nbPartitions):
        deltaWords = load("matrix/deltaWords/deltas__%04d__"%ind)
        table = np.array(list(vocabs[ind].values()))
        countWords[table[:,1]] += deltaWords[table[:, 0]]
    dump(countWords , "matrix/countWords/words_all")
    
    #Now let's update each countWords to its correct state (the old values are wrong as they
    # don't take into account the docs in other partitions)
    for ind in range(nbPartitions):
        table = np.array(list(vocabs[ind].values()))
        order = np.argsort(table[:, 0])
        table = table[order]
        dump(countWords[table[:,1]], "matrix/countWords/words__%04d__"%ind)



def getUniqueWords(ind, part):
    """Make vocabularies, a global one  and partition-specific ones.
    Each partion vacabulary is stored into disk"""
    
    vocab = np.array([])
    
    for el in part :
        vocab = np.union1d(vocab, el[1][1])
        
    return [vocab]
    

def makeVocabularies(uniqueWordsByPartition, mode = "wb"):

    vocabAll = np.unique(np.concatenate(uniqueWordsByPartition)) # Global vocabulary
    vocabAll =  {w:ind for w,ind in zip(vocabAll, range(len(vocabAll))) }
    with open("matrix/vocabulary/vocabAll", mode) as f :
        pkl.dump(vocabAll, f)
    nvocabs = len(uniqueWordsByPartition)
    for i in range(nvocabs) : # Persist partition-specific vocabularies
        v = uniqueWordsByPartition[i]
        # { word :(LocalId, GlobalId)}
        wLocIdGlobId = {w : (ind, vocabAll[w]) for w, ind in zip(v, range(len(v))) }
        with open("matrix/vocabulary/vocab__%04d__"%i, mode) as f :
            pkl.dump(wLocIdGlobId, f)
            
        print("Vocabulary %d successfully built"%i)
    print("\n Global vocabulary  built too")


def makeDocsMaps(ind,part, mode = "wb"):
    """Build a (docId, docNum) map, where docNum range from 0 to Ndocs -1. One map per partition.
    
    Let's recall that the doc partitions are disjoint"""
    
    docs = []
    
    for el in part :
        docs.append(el[1][0])
        
    partDocLocId = {doc :ind for doc, ind in zip(docs , range(len(docs))) }
    with open("matrix/docsMap/docs__%04d__"%ind, mode ) as f :
        pkl.dump(partDocLocId, f)
    return ["docMap %d successfully built"%ind]            
            

def makeConfig( **kwargs):
    id_ = kwargs["id"]
    print(id_)
    if isinstance(id_,int):
        file = "configs/config__%04d__"%id_
    else :
        file = "configs/config__all__"
#     print(json.dumps(kwargs))
    with open(file, "w") as f :
        json.dump(kwargs, f)
        
def updateConfig( **kwargs):
    id_ = kwargs["id"]
    if isinstance(id_,int):
        file = "configs/config__%04d__"%id_
    else :
        file = "configs/config__all__"
        
    with open(file, "r") as f:
        config = json.load(f)
        
    config.update(kwargs)
    
    with open(file, "w") as f :
        json.dump(config, f)



def get_now():
    return str(datetime.now())




def encode(ind, part, docs, vocabs):
    for x in part :
        el = x[1]
        yield (docs[ind][el[0]], np.array([vocabs[x[0]][w][0] for w in el[1]]) )

def loadDocsAll(nbPartitions):
    docsAll ={}
    nbDocs = []
    for  ind in range(nbPartitions):
        d = load("matrix/docsMap/docs__%04d__"%ind)
        docsAll.update(d)
        nbDocs.append(len(d))
    return docsAll


def init(corpTop,vocabs, nbDocs, nbVocabs, nvocabAll, nbTopics):
    corpTop.mapPartitionsWithIndex(saveByPartition).collect()

    corpTop.mapPartitionsWithIndex(lambda ind, part : initDocCounts(ind, part, nbDocs[ind], nbTopics)).collect()
    corpTop.mapPartitionsWithIndex(lambda ind, part : initWordCounts(ind, part, nbVocabs[ind], nbTopics)).collect()

    initCountWordsAll(vocabs, nvocabAll, nbTopics )

#     countWords = load("matrix/countWords/words_all")
#     countWords[:10]