#!/usr/bin/env python
# coding: utf-8


from multiprocessing import Process

from builder import makeConfig, updateConfig, get_now, load
from fileUtils import pickleLoader, dump
import numpy as np, time
import os, pickle as pkl, json


def pldaMap0(ind, part, alpha, beta, nbVocabAll, nbTopics):
    """Design to apply Lda in parallel on a rdd"""
    countWords = load("matrix/countWords/words__%04d__"%ind)
    countDocs = load("matrix/countDocs/docs__%04d__"%ind)
    deltaWords = np.zeros(countWords.shape)
    sumWordTopics = countWords.sum(0)
    ndocs = len(countDocs)

    for el in part :
        cDoc = countDocs[el[0]]
        tnews = []
        for w,t in zip(el[1], el[2]) :
            cDoc[t] -= 1
            deltaWords[w,t] -= 1
            countWords[w,t] -= 1
            sumWordTopics[t] -= 1
            
            if any([cDoc[t] <0, countWords[w,t] < 0, sumWordTopics[t] < 0]):
                raise ValueError("YYYYYYYYYYYYYYYY",ind, el[0],w, t, cDoc,
                                             countWords, sumWordTopics)

            proba = (cDoc + alpha)*(countWords[w] + beta)/(sumWordTopics + nbVocabAll*beta)
            if (proba<=0).any():
                raise ValueError("XXXXXXXXXXXXXXXXXX", proba.round(2), cDoc, countWords[w], sumWordTopics )
            tnew = np.random.choice(nbTopics, p =proba/proba.sum())
            
            cDoc[tnew] += 1
            deltaWords[w,tnew] += 1
            countWords[w,tnew] += 1
            sumWordTopics[tnew] += 1
            
            tnews.append(tnew)
            
        countDocs[el[0],:] = cDoc
        yield (ind, (el[0],el[1], np.array(tnews)))
    
    dump(deltaWords, "matrix/deltaWords/deltas__%04d__"%ind )
    dump(countDocs, "matrix/countDocs/docs__%04d__"%ind )





def plda_one(batchstr, countDocs, countWords,deltaWords, sumWordTopics, alpha, beta, nbVocabAll, nbTopics):
    """Design to apply  LDA a file saved as batches"""
    temp = batchstr + "__temp__"
    with open(batchstr, "rb" ) as f1 :
        with open(temp, "wb") as f2 :
            try:
                for el in pickleLoader(f1) :
                    cDoc = countDocs[el[0]]
                    tnews = []
                    for w,t in zip(el[1], el[2]) :
                        cDoc[t] -= 1
                        deltaWords[w,t] -= 1
                        countWords[w,t] -= 1
                        sumWordTopics[t] -= 1

                        if any([cDoc[t] <0, countWords[w,t] < 0, sumWordTopics[t] < 0]):
                            raise ValueError("YYYYYYYYYYYYYYYY",batchstr, el[0],w, t, cDoc[t],
                                             countWords[w,t], sumWordTopics[t])

                        proba = (cDoc + alpha)*(countWords[w] + beta)/(sumWordTopics + nbVocabAll*beta)

                        tnew = np.random.choice(nbTopics, p =proba/proba.sum())

                        cDoc[tnew] += 1
                        deltaWords[w,tnew] += 1
                        countWords[w,tnew] += 1
                        sumWordTopics[tnew] += 1

                        tnews.append(tnew)

                    countDocs[el[0],:] = cDoc
                    pkl.dump((el[0],el[1], np.array(tnews)), f2)
            except  :
#                 os.remove(temp)
                raise ValueError("************  jjjj  Something went wrong in plda_one")
                    
    os.remove(batchstr)
    os.rename(temp, batchstr)




def pldaMap(ind, nrounds, alpha, beta, nbVocabAll, nbTopics):
    """Design to apply LDA on a distributed, batched file system"""
    
    rounds = 0
    while rounds < nrounds :
        
        makeConfig(id = ind, state = "busy", time = get_now())
        
        countWords = load("matrix/countWords/words__%04d__"%ind)
        countDocs = load("matrix/countDocs/docs__%04d__"%ind)
        deltaWords = np.zeros(countWords.shape)
        sumWordTopics = countWords.sum(0)
        ndocs = len(countDocs)
        root = "matrix/corpusTopic/partition__%04d__/"%ind
        files = os.listdir(root)
        files = [file for file in files if "temp" not in file]

        for  file in files :
            plda_one(root + file, countDocs, countWords, deltaWords, sumWordTopics, alpha, beta, nbVocabAll, nbTopics)

        dump(deltaWords, "matrix/deltaWords/deltas__%04d__"%ind )
        
        rounds += 1
        
        now = get_now()
        updateConfig(id = ind, state = "free", time = now)
        
        
        timeout = 60
        poll = 1
        waited = 0
        while waited < timeout :
            with open("configs/config__all__", "r") as f:
                master = json.load(f)
                
            if master["countWordsUpdated"][str(ind)] and master["time"] >  now:
                
#                 updateConfig(id = "all",countWordsUpdated = False )
                break
            else:
                t0 = time.time()
                time.sleep(poll)
                waited += time.time() - t0
        if waited >= timeout :
            raise ValueError("Timeout : updates last too much to be done, waited %f s yet now, "%waited)



def supervise(nrounds):
    """The superviser."""
    processes = [Process(target= pldaMap, args = (ind, nrounds)) for ind in range(nbPartitions)]
    if  __name__ == "__main__":
        # Run processes
        for p in processes:
            p.start()
    count = 0        
    while count < nrounds :
        allFree = True
        for id_ in range(nbPartitions):
            with open("configs/config__%04d__"%id_, "r") as f :
                slave = json.load(f)
                if slave["state"] == "busy":
                    allFree = False
                    
        if allFree :
            updateCountWordsAll()
            updateConfig(id = "all", 
                         countWordsUpdated = {str(ind):True for ind in range(nbPartitions)}, time = get_now() )
            count += 1
        
            
        if not all([p.is_alive() for  p in processes]) :
            for p in processes :
                p.kill() 
            raise ValueError("Some process is died !!!!!!!!!!!!")
        time.sleep(1)