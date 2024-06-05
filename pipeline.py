import os
from multiprocessing import Process, Queue

from json import loads

# from time import sleep
import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd
import kblab 
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
yearCounts = {}
coWordsCache = {}

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

def addToIndex(row):
    global master_index
    tokens = tokenize(row["content"])
    idx, wordCount = text_to_idx(tokens)
    year = row["created"][:4]
    yearCounts[year] += wordCount
    for word in idx.keys():
        #if word in wordChecklist["svenskt namn"].values:
            if word in master_index:
                master_index[word][row["dark_id"], year] = idx[word]
            else:
                master_index[word] = {(row["dark_id"], year):idx[word]}

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


def getCoWordsCount(bird):
    specificMatrix = cooccurenceMatrix[bird]
    SavedCounts = {}
    for y in years:
        for x in specificMatrix[str(y)].keys():
            if x in coWordsCache:
               SavedCounts[x] = coWordsCache[x]
            if not (x in SavedCounts):
                try:
                    tmp1 = [counter(x)[str(y)]["count"] for y in years]
                    SavedCounts[x] = tmp1
                    coWordsCache[x] = tmp1
                except:
                    "not in index..."
    return SavedCounts

if __name__ == '__main__':
  yearStart = int(sys.argv[1])
  yearEnd = yearStart+10
  if yearStart == 2010:
    yearEnd +=1
  years = range(yearStart, yearEnd)

  logging.basicConfig(filename=f"{yearStart}_logs.log",
                  filemode='a',
                  format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                  datefmt='%H:%M:%S',
                  level=logging.DEBUG)
  logging.info("Program start")
  for y in years:
      yearCounts.update({str(y):0})
  totalCount = 0
  tmp = 0

  df_content: DataFrame = pd.read_feather(f"/data/birdNewsData/data/df_content_{years[0]}s.feather")
  df_numpy = np.array(df_content)
  birds = pd.read_csv(f"/data/birdNewsData/birds2.csv", header=0)
  birds = pd.read_csv("birds2.csv", header=0)
  logging.info("Loaded in the data")
  quit()

  ner = pipeline('ner', model='KB/bert-base-swedish-cased-ner', tokenizer='KB/bert-base-swedish-cased-ner')
  pos = pipeline("token-classification", model="KBLab/bert-base-swedish-cased-pos", tokenizer="KBLab/bert-base-swedish-cased-pos")
  sentiment = pipeline("text-classification", model="KBLab/robust-swedish-sentiment-multiclass")

  logging.info("Started indexing")
  df_content.apply(lambda row: addToIndex(row), axis=1)
  logging.info("Done with indexing")

  logging.info("Start calculating ")

  for i, bird in birds.iterrows():
    specificbird = bird["namn"]
    birds_dec = getDeclension(specificbird)

    cooccurenceMatrix = {}
    sentimentMatrix = {}
    for bd in birds_dec:
      logging.info(f"Start co and sen for {bd}")
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
      logging.info(f"Done co and sen for {bd}")
    
    pd.DataFrame.from_dict(cooccurenceMatrix[specificbird]).to_feather(f"/data/birdNewsData/birdFreq/df_co_{years[0]}s_{specificbird}.feather")
    specificSentiment = sentimentMatrix[specificbird]
    tmp = {}
    for y in years:
        yearlySentiment = specificSentiment[str(y)]
        countNegative = yearlySentiment.count("NEGATIVE")
        countPositive = yearlySentiment.count("POSITIVE")
        CountNeutral = yearlySentiment.count("NEUTRAL")
        tmp[y] = {"Positive": countPositive,"Neutral":CountNeutral, "Negative" : countNegative}
    pd.DataFrame.from_dict(tmp).to_feather(f"/data/birdNewsData/birdFreq/df_sen_{years[0]}s_{specificbird}.feather")

    coWordsPerBird = getCoWordsCount(specificbird)
    df_coCount = pd.DataFrame.from_dict(coWordsPerBird, orient="index", columns=years)
    df_coCount.to_feather(f"/data/birdNewsData/birdFreq/df_coWord_{years[0]}s_fågel.feather")



        



        
    
