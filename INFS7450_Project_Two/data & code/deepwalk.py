"""

@author: Yupeng Wu 45960600

Ref: tutorials by Junliang Yu
"""
import networkx as nx
from random import choice, choices
import gensim
from math import sqrt
from operator import itemgetter
import numpy as np

def readData(filename):
    data = []
    with open(filename) as f:
        for line in f:
         data.append(line.strip().split())
    return data

def evaluation(topList, groundTruth):
    count = 0
    for link in topList:
        if link[0] in groundTruth:
            count += 1
    print(float(count) / len(groundTruth) * 100, '%')

def buildGraph(data):
    G = nx.Graph()
    for pair in data:
        G.add_edge(pair[0],pair[1])
    return G

def get_bias(b_centralities, neighbors):
    bias = []
    for node in neighbors:
        bias.append(b_centralities[node])
    return bias

def randomWalk(G,length,strategy):
    if strategy == 'bfs&dfs':
        walks = []
        print('Generating dfs&bfs mixed random walks...')
        for i in range(20):
            #repeat 20 times to collect more walks for each node
            for node in G.nodes:
                #start the random walk with each node as the starter
                walk = [node]
                walkLength = 0
                while walkLength<length:
                    node = list(G.neighbors(node))[0]
                    walk.append(node)
                    node = choice(list(G.neighbors(node)))
                    walk.append(node)
                    walkLength += 1
                walks.append(walk)
        return walks
    elif strategy == 'bias':
        print('Generating betweeness centralities...')
        #length = 23
        b_centralities = nx.betweenness_centrality(G)
    walks = []
    print('Generating random walks...')
    for i in range(5):
        #repeat 5 times to collect more walks for each node
        for node in G.nodes:
            #start the random walk with each node as the starter
            walk = [node]
            walkLength = 0
            while walkLength<length:
                if strategy == 'random':
                    node = choice(list(G.neighbors(node)))
                elif strategy == 'bias':
                    neighbors = list(G.neighbors(node))
                    biases = get_bias(b_centralities, neighbors)
                    node = choices(neighbors, biases)[0]
                walk.append(node)
                walkLength += 1
            walks.append(walk)
    return walks

def skipGram(walks, method):
    print('Training Skip-Gram model... Pleas wait.')
    #using Skip-Gram
    if method == 'bfs&dfs':
        model = gensim.models.Word2Vec(walks,window=8,sg=1)
    elif method == 'bias':
        model = gensim.models.Word2Vec(walks,window=8,sg=4)
    elif method == 'random':
        model = gensim.models.Word2Vec(walks,window=5,sg=1)
    return model

def computeProximityScore(em1,em2):
    #Cosine Similarity
    score = np.dot(em1,em2)/(np.linalg.norm(em1)*np.linalg.norm(em2))
    #Euclidean distance
    #score = 1/(em1-em2).dot(em1-em2)
    return score


#read data from files
trainingData = [pair for pair in nx.read_edgelist('training.txt').edges()]
valid_pos = [pair for pair in nx.read_edgelist('val_positive.txt').edges()]
valid_neg = [pair for pair in nx.read_edgelist('val_negative.txt').edges()]
testData = [pair for pair in nx.read_edgelist('test.txt').edges()]
validationData = valid_pos+valid_neg
#build graph
G = nx.read_edgelist('training.txt')
#generate random walks
'''
# bfs&dfs miexed random walk
walks = randomWalk(G,10,'bfs&dfs')
#train model
model = skipGram(walks, 'bfs&dfs')
'''
# Try following lines for another two deep walk methods
# biased random walk (biased by betweeness centrality)
walks = randomWalk(G,20,'bias')
model = skipGram(walks, 'bias')
'''
# random walk
walks = randomWalk(G,20,'random')
model = skipGram(walks, 'random')
'''
#compute proximity score
linkScores={}
for pair in validationData:
    linkScores[pair[0]+' '+pair[1]] = computeProximityScore(model.wv[pair[0]],model.wv[pair[1]])
top100Links = sorted(linkScores.items(), key=itemgetter(1))[::-1][:100]
#validation & evaluation
groundTruth={}.fromkeys([pair[0]+' '+pair[1] for pair in valid_pos])
evaluation(top100Links,groundTruth)


#The deep walks don't have a good result, so no predictions
#prediction
linkScores={}
for pair in testData:
    linkScores[pair[0]+' '+pair[1]] = computeProximityScore(model.wv[pair[0]],model.wv[pair[1]])
top100Links = sorted(linkScores.items(), key=itemgetter(1))[::-1][:100]
#output results
results = [pair[0]+'\n' for pair in top100Links]
with open('results.txt','w') as f:
    f.writelines(results)
