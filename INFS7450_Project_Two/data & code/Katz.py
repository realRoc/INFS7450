# -*- coding: utf-8 -*-
"""

@author: Yupeng Wu 45960600

Ref: tutorials by Junliang Yu
"""

import networkx as nx
from operator import itemgetter

def evaluation(topList, groundTruth):
    count = 0
    for link in topList:
        if link[0] in groundTruth:
            count += 1
    print(float(count) / len(groundTruth) * 100, '%')

def KatzScore(u, v, G):
    a = 1
    beta = 0.5
    neighbors = list(G.neighbors(u))
    score = 0
    while a <= 1:
        path_number = neighbors.count(v)
        if path_number > 0:
            score += (beta**a)*path_number
        neighborsForNextLoop = []
        for k in neighbors:
            neighborsForNextLoop += list(G.neighbors(k))
        neighbors = neighborsForNextLoop
        a += 1
    return score

#read data from files
train_graph = nx.read_edgelist('training.txt')
valid_pos_graph = nx.read_edgelist('val_positive.txt')
valid_neg_graph = nx.read_edgelist('val_negative.txt')
test_edges = nx.read_edgelist('test.txt').edges()
valid_edges =[pair for pair in valid_pos_graph.edges()] + [pair for pair in valid_neg_graph.edges()] 
#compute Katz score
Katz = {}
for pair in valid_edges:
    Katz[pair[0] + ' ' + pair[1]] = KatzScore(pair[0], pair[1], train_graph)
top100Links = sorted(Katz.items(), key = lambda d:d[1], reverse = True)[:100]
#validation & evaluation
groundTruth={}.fromkeys([pair[0]+' '+pair[1] for pair in valid_pos_graph.edges()])
evaluation(top100Links,groundTruth)

#prediction
linkScores={}
for pair in test_edges:
    linkScores[pair[0]+' '+pair[1]] = KatzScore(pair[0], pair[1], train_graph)
top100Links = sorted(linkScores.items(), key=itemgetter(1))[::-1][:100]
#output results
results = [pair[0]+'\n' for pair in top100Links]
with open('results.txt','w') as f:
    f.writelines(results)