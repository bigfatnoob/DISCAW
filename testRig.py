from __future__ import division,print_function
import sys, random, math
import numpy as np
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LinearRegression
from lib import *
from where2 import *
import Technix.sk as sk
import Technix.CoCoMo as CoCoMo

from Models import *
MODEL = JPL.JPL

"""
Creates a generator of 1 test record 
and rest training records
""" 
def loo(dataset):
  for index,item in enumerate(dataset):
    yield item, dataset[:index]+dataset[index+1:]

"""
### Printing Stuff
Print without newline:
Courtesy @timm
"""
def say(*lst): 
  print(*lst,end="")
  sys.stdout.flush()

def formatForCART(dataset,test,trains):
  indep = lambda x: x.cells[:len(dataset.indep)]
  dep   = lambda x: x.cells[len(dataset.indep)]
  trainInputSet = []
  trainOutputSet = []
  for train in trains:
    trainInputSet+=[indep(train)]
    trainOutputSet+=[dep(train)]
  return trainInputSet, trainOutputSet, indep(test), dep(test)

"""
Selecting the closest cluster and the closest row
""" 
def clusterk1(score, duplicatedModel, tree, test, desired_effort):
  test_leaf = leaf(duplicatedModel, test, tree)
  nearest_row = closest(duplicatedModel, test, test_leaf.val)
  test_effort = effort(duplicatedModel, nearest_row)
  error = abs(desired_effort - test_effort)/desired_effort
  #print("clusterk1", test_effort, desired_effort, error)
  score += error

"""
Performing LinearRegression inside a cluster
to estimate effort
"""
def linRegressCluster(score, duplicatedModel, tree, test, desired_effort):
  
  def getTrainData(rows):
    trainIPs, trainOPs = [], []
    for row in rows:
      #trainIPs.append(row.cells[:len(duplicatedModel.indep)])
      trainIPs.append([row.cosine])
      trainOPs.append(effort(duplicatedModel, row))
    return trainIPs, trainOPs
  
  def fastMapper(test_leaf, what = lambda duplicatedModel: duplicatedModel.decisions):
    data = test_leaf.val
    one  = any(data)             
    west = furthest(duplicatedModel,one,data, what = what)  
    east = furthest(duplicatedModel,west,data, what = what)
    c    = dist(duplicatedModel,west,east, what = what)
    test_leaf.west, test_leaf.east, test_leaf.c = west, east, c
    
    for one in data:
      if c == 0:
        one.cosine = 0
        continue
      a = dist(duplicatedModel,one,west, what = what)
      b = dist(duplicatedModel,one,east, what = what)
      x = (a*a + c*c - b*b)/(2*c) # cosine rule
      one.cosine = x
      
  def getCosine(test_leaf, what = lambda duplicatedModel: duplicatedModel.decisions):
    if (test_leaf.c == 0):
      return 0
    a = dist(duplicatedModel,test,test_leaf.west, what = what)
    b = dist(duplicatedModel,test,test_leaf.east, what = what)
    return (a*a + test_leaf.c**2 - b*b)/(2*test_leaf.c) # cosine rule
    
  test_leaf = leaf(duplicatedModel, test, tree)
  fastMapper(test_leaf)
  trainIPs, trainOPs = getTrainData(test_leaf.val)
  clf = LinearRegression()
  clf.fit(trainIPs, trainOPs)
  test_effort = clf.predict(getCosine(test_leaf))
  error = abs(desired_effort - test_effort)/desired_effort
  score += error
  
  
"""
Performing LinearRegression over entire dataset
"""
def linearRegression(score, model, train, test, desired_effort):
  def getTrainData(rows):
    trainIPs, trainOPs = [], []
    for row in rows:
      trainIPs.append(row.cells[:len(model.indep)])
      trainOPs.append(effort(model, row))
    return trainIPs, trainOPs
  
  trainIPs, trainOPs = getTrainData(train)
  clf = LinearRegression()
  clf.fit(trainIPs, trainOPs)
  test_effort = clf.predict(test.cells[:len(model.indep)])
  error = abs(desired_effort - test_effort)/desired_effort
  score += error

"""
Selecting K-nearest neighbors and finding the mean
expected effort
"""
def kNearestNeighbor(score, duplicatedModel, test, desired_effort, k=1):
  nearestN = closestN(duplicatedModel, k, test, duplicatedModel._rows)
  test_effort = sorted(map(lambda x:effort(duplicatedModel, x[1]), nearestN))[k//2]
  score += abs(desired_effort - test_effort)/desired_effort  

"""
Classification and Regression Trees from sk-learn
"""
def CART(dataset, score, cartIP, test, desired_effort):
  trainIp, trainOp, testIp, testOp = formatForCART(dataset, test,cartIP);
  decTree = DecisionTreeClassifier(criterion="entropy", random_state=1)
  decTree.fit(trainIp,trainOp)
  test_effort = decTree.predict(testIp)[0]
  score += abs(desired_effort - test_effort)/desired_effort
  
def testRig(dataset=MODEL(), 
            doCART = False,doKNN = False, doLinRg = False):
  scores=dict(clstr=N(), lRgCl=N())
  if doCART:
    scores['CARTT']=N();
  if  doKNN:
    scores['knn_1'],scores['knn_3'],scores['knn_5'] = N(), N(), N()
  if doLinRg:
    scores['linRg'] = N()
  for score in scores.values():
    score.go=True
  for test, train in loo(dataset._rows):
    say(".")
    desired_effort = effort(dataset, test)
    tree = launchWhere2(dataset, rows=None, verbose=False)
    n = scores["clstr"]
    n.go and clusterk1(n, dataset, tree, test, desired_effort)
    n = scores["lRgCl"]
    n.go and linRegressCluster(n, dataset, tree, test, desired_effort)
    if doCART:
      CART(dataset, scores["CARTT"], train, test, desired_effort)
    if doKNN:
      n = scores["knn_1"]
      n.go and kNearestNeighbor(n, dataset, test, desired_effort, k=1)
      n = scores["knn_3"]
      n.go and kNearestNeighbor(n, dataset, test, desired_effort, k=3)
      n = scores["knn_5"]
      n.go and kNearestNeighbor(n, dataset, test, desired_effort, k=5)
    if doLinRg:
      n = scores["linRg"]
      n.go and linearRegression(n, dataset, train, test, desired_effort)
  return scores
  
"""
Test Rig to test CoCoMo for
a particular dataset
"""
def testCoCoMo(dataset=MODEL(), a=2.94, b=0.91):
  scores = dict(COCOMO2 = N(), COCONUT= N())
  tuned_a, tuned_b = CoCoMo.coconut(dataset, dataset._rows)
  for score in scores.values():
    score.go=True
  for row in dataset._rows:
    say('.')
    desired_effort = effort(dataset, row)
    test_effort = CoCoMo.cocomo2(dataset, row.cells, a, b)
    test_effort_tuned = CoCoMo.cocomo2(dataset, row.cells, tuned_a, tuned_b)
    scores["COCOMO2"] += abs(desired_effort - test_effort)/desired_effort
    scores["COCONUT"] += abs(desired_effort - test_effort_tuned)/desired_effort
  return scores
        
    
def testDriver():
  seed(0)
  skData = []
  dataset=MODEL()
  if  dataset._isCocomo:
    scores = testCoCoMo(dataset)
    for key, n in scores.items():
      skData.append([key+".       ."] + n.cache.all)
  scores = testRig(dataset=MODEL(),doCART = True, doKNN=True, doLinRg=True)
  for key,n in scores.items():
    if (key == "clstr" or key == "lRgCl"):
      skData.append([key+"(no tuning)"] + n.cache.all)
    else:
      skData.append([key+".         ."] + n.cache.all)

  scores = testRig(dataset=MODEL(weighFeature = True), doKNN=True)
  for key,n in scores.items():
      skData.append([key+"(sdiv_wt^1)"] + n.cache.all)
    
  print("")
  print(str(len(dataset._rows)) + " data points,  " + str(len(dataset.indep)) + " attributes")
  print("")
  sk.rdivDemo(skData)
  
testDriver()

def testKLOCWeighDriver():
  dataset = MODEL(doTune=False, weighKLOC=True)
  tuneRatio = 0.9
  skData = [];
  while(tuneRatio <= 1.2):
    dataset.tuneRatio = tuneRatio
    scores = testRig(dataset=dataset)
    for key,n in scores.items():
      skData.append([key+"( "+str(tuneRatio)+" )"] + n.cache.all)
    tuneRatio += 0.01
  print("")
  sk.rdivDemo(skData)

#testKLOCWeighDriver()

def testKLOCTuneDriver():
  tuneRatio = 0.9
  skData = [];
  while(tuneRatio <= 1.2):
    dataset = MODEL(doTune=True, weighKLOC=False, klocWt=tuneRatio)
    scores = testRig(dataset=dataset)
    for key,n in scores.items():
      skData.append([key+"( "+str(tuneRatio)+" )"] + n.cache.all)
    tuneRatio += 0.01
  print("")
  sk.rdivDemo(skData)
  
#testKLOCTuneDriver()

#testRig(dataset=MODEL(doTune=False, weighKLOC=False), duplicator=interpolateNTimes)
