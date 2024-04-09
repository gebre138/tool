try:
    from IPython.display import clear_output
    clear_output(wait=True)
except ImportError as err:
    print(err)
try:
    import os
    import pandas as pd
    import glob
    import string
    import operator
    import pydot
    import re
    import math
    import hm
    import numpy as np
    from termcolor import colored
    from gensim.models import Word2Vec, KeyedVectors
    import matplotlib.pyplot as plt
    from keras.models import load_model
    from collections import Counter
    import base64
    import requests
    import sys
    from matplotlib import pyplot as plt    
except ImportError as err:
    print(err)

def corpus_build(corpus):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
    path = os.getcwd()
    paths="/content/drive/MyDrive/DataAnnotation" if path=="/content" else path
    def internet_connection():
        try:
            response = requests.get("https://raw.githubusercontent.com/gebre138/annotation/main/stoplist/spchar.txt", timeout=5)
            return True
        except requests.ConnectionError:
            return False    
    if internet_connection():
        spch=requests.get("https://raw.githubusercontent.com/gebre138/annotation/main/stoplist/spchar.txt").text.split()
    else:
        print("Connect to the internet")
        sys.exit()
    #********SENTENCE ROOT EXTRUCTION****************
    text=re.split('[?።!\n]', corpus)
    proscorpus=""
    for sent in text:
        rootsent=""
        for words in sent.split():
            if words not in spch:
                wordrt=hm.anal('amh', words, um=True)
                if wordrt!=[]:
                    wordlema=wordrt[0]['lemma'].replace("|", "/")
                    if "/" in wordlema:
                        reslt = re.search('(.*)/', wordlema)
                        rootsent+=reslt.group(1)+" "
                    else:
                        rootsent=rootsent+" "+wordlema+" "
                else:
                    rootsent=rootsent+" "+words+" "
            else:
                rootsent=rootsent+" "+words+" "
        rootsent=rootsent.strip()
        proscorpus=proscorpus+rootsent+"\n"
    proscorpus=proscorpus.strip()
    if os.path.exists(paths+"/dataset/corpus.txt"):
        proscorpus=proscorpus+open(paths+"/dataset/corpus.txt",'r',encoding="utf-8-sig").read()
        os.remove(paths+"/dataset/corpus.txt")
    with open(paths+'/dataset/corpus.txt', 'a',encoding="utf-8-sig") as file:
        file.write(str(proscorpus)+"\n")
    #******EXTRUCT WORD WITH ITS FREQUENCY***********
    dect=Counter(corpus.split())
    dect_val=list(map(lambda x: round(x/len(dect),5), list(dect.values())))
    word_with_freqcy=dict(zip(list(dect.keys()), dect_val))
    if os.path.exists(paths+"/dataset/word_frequency.txt"):
        os.remove(paths+"/dataset/word_frequency.txt")
    with open(paths+'/dataset/word_frequency.txt', 'a',encoding="utf-8-sig") as file:
        file.write(str(word_with_freqcy)+"\n")
    #*******EXTRUCT SENTENCE FREQUENCY*****************
    sentsum=0
    sentdic={}
    for snts in list(filter(None,corpus.splitlines())):
        sentsum=sum([word_with_freqcy[w] for w in snts.split()])
        sentdic.update({snts:round(sentsum/len(snts.split()),5)})
    sentdic=sorted(sentdic.items(), key=lambda x:x[1])
    #*******SAVE SENTENCE WITH ITS FREQUENCY*****************
    if os.path.exists(paths+"/dataset/sentence_frequency.txt"):
        sentdic=str(sentdic)+open(paths+"/dataset/sentence_frequency.txt",'r',encoding="utf-8-sig").read()
        os.remove(paths+"/dataset/sentence_frequency.txt")
    with open(paths+'/dataset/sentence_frequency.txt', 'a',encoding="utf-8-sig") as file:
        file.write(str(sentdic)+"\n")
    #*******FIND COMPLEXITY RANGE***************** 
    print("Total Vocabulary: "+str(len(proscorpus.split()))+"\n"+"Dectionary size: "+str(len(dect)))
    mid=(sentdic[0][1]+sentdic[len(sentdic)-1][1])/2
    lowmid=(sentdic[0][1]+mid)/2
    if os.path.exists(paths+"/dataset/sentence_complexity_range.txt"):
        os.remove(paths+"/dataset/sentence_complexity_range.txt")
    with open(paths+'/dataset/sentence_complexity_range.txt', 'a',encoding="utf-8") as file:
        file.write(str(round(mid,5))+"\n"+str(round(lowmid,5)))

def ComplexityAnotator(text):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
    path = os.getcwd()
    paths="/content/drive/MyDrive/DataAnnotation" if path=="/content" else path
    os.makedirs(paths+"/dataset") if not os.path.exists(paths+"/dataset") else ''
    os.makedirs(paths+"/dataset/other") if not os.path.exists(paths+"/dataset/other") else ''
    def internet_connection():
        try:
            response = requests.get("https://raw.githubusercontent.com/gebre138/annotation/main/stoplist/spchar.txt", timeout=5)
            return True
        except requests.ConnectionError:
            return False    
    if internet_connection():
        spch=requests.get("https://raw.githubusercontent.com/gebre138/annotation/main/stoplist/spchar.txt").text.split()
        amharicstop=requests.get("https://raw.githubusercontent.com/gebre138/annotation/main/stoplist/amharic_stop_lists.txt").text.split()
        corpus=requests.get("https://raw.githubusercontent.com/gebre138/annotation/main/corpus.txt").text.split()
        sfreq=requests.get("https://raw.githubusercontent.com/gebre138/annotation/main/sentence_complexity_range.txt").text.split()
    else:
        print("Connect to the internet")
        sys.exit()

    column=["text","complexw","comwvgfreq","sentvgfreq","label"]
    lemma = pd.DataFrame(columns=["word","count"])
    dataset = pd.DataFrame(columns=column)
    complx = pd.DataFrame(columns=column)
    noncomplx = pd.DataFrame(columns=column)
    preproces = pd.DataFrame(columns=column)
    #*****BUILD COMPLEX TERMS******************
    if not os.path.exists(paths+"/dataset/other/complex_word.xlsx"):
        root=requests.get("https://raw.githubusercontent.com/gebre138/annotation/main/stoplist/complex_root_word.txt").text.split()
        for w in root:
            lemma.loc[len(lemma.index)]=[w,0]
        lemma.to_excel(paths+'/dataset/other/complex_word.xlsx',index=False)
    #********SAVE NEW DATASET**********************
    if os.path.exists(paths+"/dataset/dataset.xlsx"):
        complxold = pd.read_excel(paths+"/dataset/dataset.xlsx")
        os.remove(paths+"/dataset/dataset.xlsx")
        complx=pd.concat([complxold,complx])
    #********SAVE RESERVED ANNOTATED DATA***********
    if os.path.exists(paths+"/dataset/other/reserve.xlsx"):
        reserves = pd.read_excel(paths+"/dataset/other/reserve.xlsx")
        os.remove(paths+"/dataset/other/reserve.xlsx")
        noncomplx=pd.concat([reserves,noncomplx])
    #******EXTRUCT SENTENCE AND ITS ROOT************
    sentences=re.split('[?።!\n]', text)
    with open(paths+'/dataset/sentence.txt', 'a',encoding="utf-8") as file:
        file.write("\n".join(sentences)+"\n")
    if os.path.exists(paths+"/dataset/sentence.txt"):
        allfiles = glob.glob(paths+'/dataset/sentence.txt')#most change simple to sentence
        df = pd.concat((pd.read_csv(f, header=None, names=["text"]) for f in allfiles))
        lemma = pd.read_excel(paths+"/dataset/other/complex_word.xlsx")
        if df.empty==False:
            for sent in df["text"]:
                rootsent=""
                for words in sent.split():
                    if words not in spch:
                        wordrt=hm.anal('amh', words, um=True)
                        if wordrt!=[]:
                            wordlema=wordrt[0]['lemma'].replace("|", "/")
                            if "/" in wordlema:
                                reslt = re.search('(.*)/', wordlema)
                                rootsent+=reslt.group(1)+" "
                            else:
                                rootsent=rootsent+" "+wordlema+" "
                        else:
                            rootsent=rootsent+" "+words+" "
                    else:
                        rootsent=rootsent+" "+words+" "
     #********ANNOTATE THE SENTENCE**************************
                avgsentfreq=0
                if (set(lemma["word"]) & set(rootsent.split())) and sent not in complx.text.values:
                    comp=set(lemma["word"]) & set(rootsent.split())
                    avgwfreq=sum([round(corpus.count(w)/len(Counter(corpus)),5) for w in comp])/len(comp)
                    avg_s_f=round(sum([round(corpus.count(w)/len(Counter(corpus)),5) for w in rootsent.split()])/len(rootsent.split()),5)
                    for i in list(comp):
                        lemma.loc[lemma[lemma['word']==i].index.values,'count']=lemma.loc[lemma[lemma['word']==i].index.values,'count']+1
                    clear_output(wait=True)
                    print(sent)
                    if avg_s_f<=float(sfreq[0]):
                        complx.loc[len(complx)]=[sent,', '.join(comp),avgwfreq,avg_s_f,3]
                    elif avg_s_f>float(sfreq[0]) and avg_s_f<=float(sfreq[1]):
                        complx.loc[len(complx)]=[sent,', '.join(comp),avgwfreq,avg_s_f,2]
                    else:
                        complx.loc[len(complx)]=[sent,', '.join(comp),avgwfreq,avg_s_f,1]
                elif sent not in complx.text.values and (complx['label']!=0).sum() > (complx['label']==0).sum():
                    complx.loc[len(complx)]=[sent,"-","-",1,0]
                elif sent not in complx.text.values and sent not in noncomplx.text.values:
                    noncomplx.loc[len(noncomplx)]=[sent,"-","-",1,0]
    #**********SAVE NEW COMPLEX TERMS AS EXCEL******************    
    os.remove(paths+"/dataset/other/complex_word.xlsx") if os.path.exists(paths+"/dataset/other/complex_word.xlsx") else ''
    lemma.to_excel(paths+'/dataset/other/complex_word.xlsx',index=False)
    #**********BALANCE ANNOTATED DATASET***********************    
    result=(complx['label']!=0).sum()-(complx['label']==0).sum()
    if noncomplx.empty==False:
        for indx, txt in enumerate(noncomplx["text"]):
            if result > indx and txt not in complx.text.values:
                complx.loc[len(complx)]=[txt,"-","-",1,0] 
            else: 
                break
        noncomplx.drop(noncomplx.index[0:indx], inplace=True)
    #*******SAVE NEW DATASET********************************
    os.remove(paths+"/dataset/sentence.txt") if os.path.exists(paths+"/dataset/sentence.txt") else ''
    complx.to_excel(paths+'/dataset/dataset.xlsx',index=False)
    noncomplx.to_excel(paths+'/dataset/other/reserve.xlsx',index=False) 
    #*********8PRINT DATA CHART******************************
    Aus_Players = 'High Complex', 'Mid Level Complex', 'Low Level Complex', 'Non-Complex'    
    Runs = [(complx['label']==3).sum(), (complx['label']==2).sum(), (complx['label']==1).sum(), (complx['label']==0).sum()]    
    explode = (0, 0, 0, 0.1)  
    if (complx['label']!=0).sum()!=0:
        fig1, ax1 = plt.subplots()    
        ax1.pie(Runs, explode=explode, labels=Aus_Players, autopct='%1.1f%%',    
                shadow=True, startangle=90)    
        ax1.axis('equal')    
        plt.show()   
    else:
        print("No Complex data found in corpus")
    print("Annotated Data: " + str(len(complx))+"\n"+"Please find the Dataset in path: "+paths+'/dataset/dataset.xlsx'+"\n")