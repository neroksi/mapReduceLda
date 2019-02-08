#!/usr/bin/env python
# coding: utf-8



import pickle as pkl
import os, shutil


def load(path):
    with open(path, "rb") as f :
        obj = pkl.load(f)
    return obj


def dump(obj,path, mode = "wb"):
    with open(path, mode) as f :
        pkl.dump(obj,f)



def saveByPartition(ind, part, folder = "pickle", mode = "wb", batchsize = 10):
    """Save a rdd, mappped with partition index into disk. """
    
    root = folder + "/partition__%04d__"%ind # Root file by partition

    if os.path.exists(root):
        if mode == "wb":
            shutil.rmtree(root) # remove file if it exists and in overwrite mode
    os.mkdir(root) # Create dir
    write_more = True # Are we at the end of the partition ?
    counter = 0 # Batch num
    while write_more :
        write_more = False
        file = root + "/batch_%010d"%counter
        nwrited = 0 # writed line number
        with open(file, mode ) as f :
            for el in part :
                pkl.dump(el, f)
                nwrited += 1
                write_more = True
                if nwrited >= batchsize :
                    break
            
            counter += 1
        if not write_more :
            os.remove(file)
            break
    return [] # return empty list, as the mapPartitionWithIndex requires an iterable to be returned


def getDocsAll():
    docsAll ={}
    nbDocs = []
    for  ind in range(nbPartitions):
        d = load("matrix/docsMap/docs__%04d__"%ind)
        docsAll.update(d)
        nbDocs.append(len(d))


def pickleLoader(pklFile):
    try:
        while True:
            yield pkl.load(pklFile)
    except EOFError:
        pass
    
    
def saveAsPickleFile(rdd):
    if os.path.exists("pickle_old/") :
          shutil.rmtree("pickle_old/")
    rdd.saveAsPickleFile("pickle_old")
    shutil.rmtree("pickle/")
    os.renames("pickle_old/", "pickle")