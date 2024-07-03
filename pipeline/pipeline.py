import os
from multiprocessing import Process, Queue

from json import loads

# from time import sleep

#from cuml.cluster import HDBSCAN
#from cuml.manifold import UMAP
#from cuml.preprocessing import normalize

from tqdm import tqdm
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import logging
import numpy as np
from umap import UMAP
from bertopic.vectorizers import ClassTfidfTransformer
import torch
import pandas as pd
import datamapplot
import kblab 
from bertopic import BERTopic
import sys
import math
from pandas.core.frame import DataFrame
from tqdm import tqdm
from urllib3.util import Retry
from urllib3 import PoolManager, make_headers
from kblab import Archive
import regex as re
from itertools import product
kblab.VERIFY_CA=False
from transformers import AutoModel,pipeline

master_index = {}
docs = []
filtered_count_index = {}
yearCounts = {}

regex = r"\p{L}+"
def tokenize(text):
  return re.finditer(regex, text.lower())

def text_to_idx(words):
  wordCount = 0
  wordPos = {}
  for token in words:
    wordCount+=1
    word = token.group()
    pos = token.span()[0]
    if word in wordPos.keys():
      wordPos[word].append(pos)
    else:
      wordPos[word] = [pos]
  return wordPos, wordCount

def getDeclension(word):
  if word[-2:] == "ås":
    root = word[:-2]
    return [word, f"{word}en", f"{root}äss", f"{root}ässen"]
  if word[-2:] == "tt" or word[-2:] == "st":
    root = word[:-1]
    return [word, f"{word}en", f"{word}ar", f"{word}arna"]
  if word[-2:] == "rt":
    root = word[:-1]
    return [word, f"{word}en", f"{word}ar", f"{word}arna"]
  if word[-1:] == "t":
    root = word[:-1]
    return [word, f"{word}en", f"{word}er", f"{word}erna"]
  if word[-3:] == "gam":
    return [word, f"{word}en", f"{word}ar", f"{word}arna"]
  if word[-1:] == "m":
    return [word, f"{word}men", f"{word}mar", f"{word}marna"]
  if word[-2:] == "ss":
    return [word, f"{word}en", f"{word}er", f"{word}erna"]
  if word[-1:] == "a":
    root = word[:-1]
    return [word, f"{root}an", f"{root}or", f"{root}orna"]
  if word[-2:] == "yr":
    root = word[:-2]
    return [word, f"{word}en", f"{root}rar", f"{root}rarna"]
  if word[-3:] == "ger":
    root = word[:-2]
    return [word, f"{root}ern", f"{root}rar", f"{root}rarna"]
  if word[-1:] == "r" and (not word[-2:] == "är"):
    root = word[:-1]
    return [word, f"{word}et", f"{word}", f"{word}en"]
  if word[-1:] == "d":
    root = word[:-3]
    return [word, f"{word}et", f"{root}änder", f"{root}änderna"]
  if word[-3:] == "are" and (not word[-5:] == "stare"):
    root = word[:-1]
    return [word, f"{root}en", f"{root}e", f"{root}na"]
  if word[-2:] == "re":
    root = word[:-1]
    return [word, f"{root}en", f"{root}ar", f"{root}arna"]
  if word[-1:] == "e":
    root = word[:-1]
    return [word, f"{root}en", f"{root}ar", f"{root}arna"]
  if word[-4:] == "rell":
    root = word[:-2] 
    return [word, f"{word}en", f"{word}er", f"{word}erna"]
  if word[-2:] == "ll":
    root = word[:-2] 
    return [word, f"{word}en", f"{word}ar", f"{word}arna"]
  if word[-2:] == "yl" or word[-3:] == "nal":
    root = word[:-2] 
    return [word, f"{word}en", f"{word}er", f"{word}erna"]
  if word[-3:] == "gal":
    root = word[:-2] 
    return [word, f"{word}en", f"{word}ar", f"{word}arna"]
  if word[-1:] == "l":
    root = word[:-2] 
    return [word, f"{root}eln", f"{root}lar", f"{root}larna"]
  if word[-2:] == "ag":
    root = word 
    return [word, f"{root}et", f"{root}", f"{root}en"]
  if word[-1:] == "g":
    root = word 
    return [word, f"{root}en", f"{root}ar", f"{root}arna"]
  if word[-2:] == "um":
    root = word[:-2] 
    return [word, f"{root}en", f"{root}nar", f"{root}narna"]
  if word[-2:] == "en":
    root = word[:-2] 
    return [word, f"{root}nen", f"{root}er", f"{root}erna"]
  if word[-2:] == "an":
    root = word[:-2] 
    return [word, f"{word}en", f"{word}er", f"{word}erna"]
  if word[-2:] == "ur" or word[-2:] == "är":
    root = word[:-2] 
    return [word, f"{word}en", f"{word}er", f"{word}erna"]
  if word[-2:] == "nd":
    root = word[:-3]
    return [word, f"{word}en", f"{root}änder", f"{word}änderna"]
  if word[-1:] == "ö":
    return [word, f"{word}n", f"{word}ar", f"{word}arna"]
  if word[-2:] == "es":
    return [word, f"{word}en", f"{word}ar", f"{word}arna"]
  if word[-2:] == "rk" or word[-2:] == "nk":
    return [word, f"{word}en", f"{word}ar", f"{word}arna"]
  if word[-2:] == "ök" or word[-2:] == "åk" or word[-2:] == "lk" or word[-2:] == "ck":
    return [word, f"{word}en", f"{word}ar", f"{word}arna"]
  if word[-1:] == "k":
    return [word, f"{word}en", f"{word}er", f"{word}erna"]
  if word[-1:] == "o":
    return [word, f"{word}n", f"{word}er", f"{word}rna"]
  if word[-1:] == "s" and (not word[-2:] == "is" or not word[-2] == "es"):
    return [word, f"{word}et", f"{word}", f"{word}en"]
  else:
    root = word
    return [word, f"{root}en", f"{root}ar", f"{root}arna"]


def addWords(word, onlyDeclension = True): 
    global wordChecklist
    if onlyDeclension:
        wordDec = getDeclension(word)[1:]
        wordDecFrame = pd.DataFrame([wordDec[0],wordDec[1],wordDec[2]], columns=["naturtyp"])
    else:
        wordDec = getDeclension(word)
        wordDecFrame = pd.DataFrame([wordDec[0],wordDec[1],wordDec[2],wordDec[3]], columns=["naturtyp"])
    wordChecklist = pd.concat([wordChecklist,wordDecFrame])

def concatDictDF(dict, pastDf):
    return pastDf.combine_first(pd.DataFrame.from_dict(dict))

def addToIndex(row, saveIndex):
    tmp = 0
    global master_index
    global filtered_count_index
    tokens = tokenize(row["content"])
    idx, wordCount = text_to_idx(tokens)
    year = row["created"][:4]
    yearCounts[year] += wordCount
    for word in idx.keys():
        if saveIndex == 1:
          boolWord = word in wordChecklist["naturtyp"].values
        if word in master_index:
            master_index[word][row["dark_id"], year] = idx[word]
            if saveIndex == 1:
              if boolWord:
                filtered_count_index[word][row["dark_id"]] = len(idx[word])
        else:
            master_index[word] = {(row["dark_id"], year):idx[word]}
            if saveIndex == 1:
              if boolWord:
                filtered_count_index[word] = {row["dark_id"]:len(idx[word])}

#        if saveIndex == 1:
#          if word in wordChecklist["naturtyp"].values:
#                filtered_entry = {word:{row["dark_id"]:len(idx[word])}}        
#                if word in filtered_count_index:               
#                else:
                  #master_index[word] = {(row["dark_id"], year):idx[word]}

def counter(word):
  master_index_bird = master_index[word]
  count = {}

  for y in years:
    count.update({str(y): {"freq":0, "count":0, "prob":0}})
  for a,b in master_index_bird:
    if b in count:
      count[b]["count"] += 1
    else:
      count[b]["count"] = 1
  for y in years:
    try:
      count[str(y)]["freq"] = count[str(y)]["count"]/yearCounts[str(y)]*100000
      count[str(y)]["prob"] = count[str(y)]["count"]/yearCounts[str(y)]
    except:
      count[str(y)]["freq"] = 0.0
  return count

def getWordCooccurenceAndSentiment(word):
  yearsString = [str(y) for y in years]
  master_index_bird = master_index[word]
  sentimentByYear = {}
  global df_numpy
  cooccurence = {}
  for y in years:
    cooccurence.update({str(y):{}})
    sentimentByYear.update({str(y):[]})
  for issue, year in master_index_bird:
    rows, cols = np.where(df_numpy == issue)
    textBlock = df_numpy[rows][0][0].lower().replace(",", "")
    n = 5
    lhs, bird, rhs = textBlock.partition(word)
    window = lhs.split()[-n:] + rhs.split()[:n]
    sentimentWindow = " ".join(lhs.split()[-n:] + [bird] + rhs.split()[:n])
    if len(sentimentWindow) > 256:
      sentimentWindow = lhs[-128:] + bird + rhs[128:]
    textSentiment = sentiment(sentimentWindow)[0]["label"]
    sentimentByYear[year].append(textSentiment)
    birdWords = list(filter(None, window))
    for bWord in birdWords:
      if bWord == word or bWord in yearsString:
        continue
      if bWord in cooccurence[year]:
        cooccurence[year][bWord] += 1
      else:
        cooccurence[year][bWord] = 1  
  return cooccurence, sentimentByYear

def mergeSentiment(s1,s2):
    tmp = {}
    for y in years:
        tmp1 = s1[str(y)]
        tmp2 = s2[str(y)]
        tmp[str(y)] = tmp1+tmp2
    return tmp

def mergeCounts(c1, c2):
  tmp = {}
  for y in years:
    tmp[str(y)] = {"freq": c1[str(y)]["freq"] + c2[str(y)]["freq"], "count": c1[str(y)]["count"] + c2[str(y)]["count"], "prob": c1[str(y)]["prob"] + c2[str(y)]["prob"]}
  return tmp

def mergeCoocs(c1, c2):
  tmp = {}
  for y in years:
    tmp[str(y)] = {k: (c1[str(y)].get(k, 0) + c2[str(y)].get(k, 0)) for k in set(c1[str(y)]) | set(c2[str(y)])}
  return tmp

def indices(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset+1)
        except ValueError:
            return result
        result.append(offset)

def getDocs(word):
    global docs
    master_index_bird = master_index[word]
    for issue, year in master_index_bird:
        res = df_content.loc[df_content["dark_id"] == issue]
        res = res["content"].values
        textBlock = res[0].lower().replace(",", "")
        n = 10
        idxs = indices(textBlock, word)
        for id in idxs:
            lhs, bird, rhs = textBlock[id-50:id+50].partition(word)
            sentimentWindow = " ".join(lhs.split()[-n:] + [bird] + rhs.split()[:n])
            docs.append(sentimentWindow)

def getCoWordsCount(bird):
    specificMatrix = cooccurenceMatrix[bird]
    SavedCounts = {}
    for y in years:
        for x in specificMatrix[str(y)].keys():
            if not (x in SavedCounts):
                try:
                    tmp1 = [counter(x)[str(y)]["count"] for y in years]
                    SavedCounts[x] = tmp1
                except:
                    "not in index..."
    return SavedCounts

if __name__ == '__main__':
  
  yearStart = int(sys.argv[1])
  birdOption = int(sys.argv[2])
  saveIndex = int(sys.argv[3])
  trainModel = int(sys.argv[4])
  print(torch.cuda.is_available())

  logging.basicConfig(filename=f"secondTrip_{yearStart}.log",
                  filemode='a',
                  format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                  datefmt='%H:%M:%S',
                  level=logging.DEBUG)
  logging.info("Program start")

  saveName = "na"
  if birdOption == 0:
    logging.info("Small bird list selected")
    birds = pd.read_csv("../birds2.csv", header=0)
    saveName = "small"
  elif birdOption == 1:
    logging.info("Dialectic bird list selected")
    birds = pd.read_csv("../birdsDialectic.csv")
    saveName = "dialect"
  else:
    logging.info("Other words selected")
    dfLan = pd.read_csv("csvFiles/lan.csv")
    dfLander = pd.read_csv("csvFiles/lander.csv")
    dfman = pd.read_csv("csvFiles/manader.csv")
    dfnat = pd.read_csv("csvFiles/nationaliteter.csv")
    dfort = pd.read_csv("csvFiles/ortsnamn.csv")
    dfVecko = pd.read_csv("csvFiles/veckodagar.csv")
    dfVoc = pd.read_csv("csvFiles/vocations.csv")
    birds = pd.concat([dfLan,dfLander,dfman,dfnat,dfort,dfVecko,dfVoc])
    saveName = "others"


  yearEnd = yearStart+10
  if yearStart == 2010:
    yearEnd +=1
  years = range(yearStart, yearEnd)


  for y in years:
      yearCounts.update({str(y):0})
  totalCount = 0
  tmp = 0

  logging.info(f"options: {sys.argv}")

  df_content: DataFrame = pd.read_feather(f"/data/birdNewsData/data/df_content_{years[0]}s.feather")
  df_numpy = np.array(df_content)

  

  logging.info("Loaded in the data")

  """ ner = pipeline('ner', model='KB/bert-base-swedish-cased-ner', tokenizer='KB/bert-base-swedish-cased-ner')
  pos = pipeline("token-classification", model="KBLab/bert-base-swedish-cased-pos", tokenizer="KBLab/bert-base-swedish-cased-pos")
  sentiment = pipeline("text-classification", model="KBLab/robust-swedish-sentiment-multiclass") """

  natureTypes = pd.read_csv("../naturtyper.csv", header=0)
  wordChecklist = natureTypes.copy()
  natureTypes["naturtyp"].values

  for x in natureTypes["naturtyp"].values:
    addWords(x)

  for x in birds["namn"].values:
    addWords(x, False)
  
  logging.info("Started indexing")
  logging.info(f"Save index: {saveIndex}")

  tqdm.pandas()

  for y in years:
    if trainModel == 1:
      df_content[df_content["created"].str.contains(str(y))].sample(frac=0.01).progress_apply(lambda row: addToIndex(row, saveIndex), axis=1)
    else:
      df_content[df_content["created"].str.contains(str(y))].sample(frac=0.10).progress_apply(lambda row: addToIndex(row, saveIndex), axis=1)

  
  if saveIndex == 1:
    logging.info("Saving index")
    dfFilteredTmp = pd.DataFrame.from_dict(filtered_count_index)
    dfFilteredTmp.to_feather(f"/data/birdNewsData/secondTrip/df_filtered_{saveName}_index_{years[0]}s.feather")
  logging.info("Done with indexing")

  logging.info("Start calculating ")

  frequency = {}

  for i, bird in birds.iterrows():
    specificbird = bird["namn"]
    birds_dec = getDeclension(specificbird)
    for bd in birds_dec:
      if birdOption == 0:
        if trainModel == 1:
          try:
            getDocs(bd)
          except:
            "not found, is ok"
      try:
        bc = counter(bd)
        if bird[0] in frequency:
          freq = frequency[specificbird]
        else:
          freq = {}
          for y in years:
            freq.update({str(y):{"freq":0, "count":0, "prob":0}})
        frequency[specificbird] = mergeCounts(bc, freq)
      except:
        "bird not found"
  logging.info("Done with calculating") 
  dataFreq = {}
  dataCount = {}

  for b in frequency.keys():
    dataFreq.update({b: [frequency[b][x]["freq"] for x in frequency[b]]})
    dataCount.update({b: [frequency[b][x]["count"] for x in frequency[b]]})

  dfFreq = pd.DataFrame.from_dict(dataFreq, orient="index", columns=years)
  dfCount = pd.DataFrame.from_dict(dataCount, orient="index", columns=years)

  if trainModel == 0:
    dfFreq.to_feather(f"/data/birdNewsData/secondTrip/dfFreq_sampled_{saveName}_{years[0]}s.feather")
    dfCount.to_feather(f"/data/birdNewsData/secondTrip/dfCount_sampled_{saveName}_{years[0]}s.feather")
    logging.info("Written freq and count to file")

  if trainModel == 1:
    #umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0)
    #hdbscan_model = HDBSCAN(min_samples=10, gen_min_span_tree=True, prediction_data=True)
    

    logging.info("Starting building bertopic model")
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    embedding_model = 'KBLab/sentence-bert-swedish-cased'
    sentence_model = SentenceTransformer(embedding_model)
    embeddings = sentence_model.encode(docs, show_progress_bar=True)
    #embeddings = normalize(embeddings)
    topic_model = BERTopic(verbose=True, ctfidf_model=ctfidf_model)
#    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model,verbose=True, ctfidf_model=ctfidf_model)
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
    logging.info("Done building bertopic model")
    topic_model.__dict__.pop("representative_docs_")
    topic_model.save(f"ownData/{saveName}_{years[0]}s", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)
    logging.info("Saved Bertopic model")
    #logging.info("Reduce embeddings")
    #reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    #logging.info("Done reducing embeddings")
    #topic_model.reduce_topics(docs, nr_topics=50)
    #logging.info("Creating datamapplot on 50 topics")
    #fig = topic_model.visualize_document_datamap(docs, reduced_embeddings=reduced_embeddings)
    #fig.savefig(f"figures/{saveName}_{years[0]}s.png", bbox_inches="tight")
    #try:
    #  topic_model.__dict__.pop("representative_docs_")
    #except:
    #  "popped already"
    #topic_model.save(f"ownData/reduced_{saveName}_{years[0]}s", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)

    
    


  logging.info(f"{years[0]}s for {saveName} done")

  """ for i, bird in birds.iterrows():
    specificbird = bird["namn"]
    birds_dec = getDeclension(specificbird)

    cooccurenceMatrix = {}
    sentimentMatrix = {}
    for bd in birds_dec:
      logging.info(f"Start co and sen for {bd}")
      try:
        coWord, senWord = getWordCooccurenceAndSentiment(bd)
        if specificbird in cooccurenceMatrix:
            co = cooccurenceMatrix[specificbird]
            cooccurenceMatrix[specificbird] = mergeCoocs(co, coWord)
        else:
            cooccurenceMatrix[specificbird] = coWord

        if specificbird in sentimentMatrix:
            se = sentimentMatrix[specificbird]
            sentimentMatrix[specificbird] = mergeSentiment(se, senWord)
        else:
            sentimentMatrix[specificbird] = senWord
      except:
        logging.info(f"Abandoned co and sen for {bd}, not found in index")
      logging.info(f"Done co and sen for {bd}")
    
    try:
      pd.DataFrame.from_dict(cooccurenceMatrix[specificbird]).to_feather(f"/data/birdNewsData/birdFreq/df_co_{years[0]}s_{specificbird}.feather")
      logging.info(f"Written to file /data/birdNewsData/birdFreq/df_co_{years[0]}s_{specificbird}.feather")
      specificSentiment = sentimentMatrix[specificbird]
      tmp = {}
      for y in years:
          yearlySentiment = specificSentiment[str(y)]
          countNegative = yearlySentiment.count("NEGATIVE")
          countPositive = yearlySentiment.count("POSITIVE")
          CountNeutral = yearlySentiment.count("NEUTRAL")
          tmp[y] = {"Positive": countPositive,"Neutral":CountNeutral, "Negative" : countNegative}
      pd.DataFrame.from_dict(tmp).to_feather(f"/data/birdNewsData/birdFreq/df_sen_{years[0]}s_{specificbird}.feather")
      logging.info(f"Written to file /data/birdNewsData/birdFreq/df_sen_{years[0]}s_{specificbird}.feather")

      coWordsPerBird = getCoWordsCount(specificbird)
      df_coCount = pd.DataFrame.from_dict(coWordsPerBird, orient="index", columns=years)
      df_coCount.to_feather(f"/data/birdNewsData/birdFreq/df_coWord_{years[0]}s_{specificbird}.feather")
      logging.info(f"Written to file /data/birdNewsData/birdFreq/df_coWord_{years[0]}s_{specificbird}.feather")
    except:
      logging.info(f"{specificbird} not found in decade") """