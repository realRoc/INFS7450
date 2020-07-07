from scipy.sparse import coo_matrix
import numpy as np
from collections import defaultdict
from operator import itemgetter

Epsilon = 10e-5

def readData(filename):
    data = []
    with open(filename) as f:
        for line in f:
         data.append(line.strip().split())
    return data

def getNeighbors(data):
    neighbors=defaultdict(set)
    for pair in data:
        neighbors[pair[0]].add(pair[1])
        neighbors[pair[1]].add(pair[0])
    return neighbors

def buildIndex(data):
    index = {}
    for pair in data:
        if pair[0] not in index:
            index[pair[0]] = len(index)
        if pair[1] not in index:
            index[pair[1]] = len(index)
    return index

def buildSparseAdjacencyMatrix(data,index,neighbors):
    row, col, entries = [],[],[]
    for pair in data:
        #symmetric matrix
        row += [index[pair[0]],index[pair[1]]]
        col += [index[pair[1]],index[pair[0]]]
        entries += [1,1]
    AdjacencyMatrix = coo_matrix((entries,(row,col)),shape=(len(index),len(index)))

    row, col, entries = [], [], []
    for node in neighbors:
        row.append(index[node])
        col.append(index[node])
        entries.append(1.0/len(neighbors[node]))
    DegreeMatrix = coo_matrix((entries,(row,col)),shape=(len(index),len(index)))
    return AdjacencyMatrix.dot(DegreeMatrix)

def personalizedPageRank(restart,damper,ajacencyMatrix):
    size = ajacencyMatrix.shape[0]
    r = np.ones(size)/size
    r_ = np.zeros(size)

    while (r-r_).dot(r-r_) > Epsilon:
        r_ = r
        r = damper*ajacencyMatrix.dot(r)
        r[restart] += (1-damper)
    return r

def evaluation(topList, groundTruth):
    count = 0
    for link in topList:
        if link[0] in groundTruth:
            count += 1
    print(float(count) / len(groundTruth) * 100, '%')

def main():
    #read data from files
    trainingData = readData('training.txt')
    valid_pos = readData('val_positive.txt')
    valid_neg = readData('val_negative.txt')
    testData = readData('test.txt')
    validationData = valid_pos+valid_neg
    #data preparation
    user_in_validation_set = set()
    for pair in validationData:
        user_in_validation_set.add(pair[0])
        user_in_validation_set.add(pair[1])
    user_in_test_set = set()
    for pair in testData:
        user_in_test_set.add(pair[0])
        user_in_test_set.add(pair[1])
    neighbors = getNeighbors(trainingData)
    index = buildIndex(trainingData)
    AjacencyMatrix = buildSparseAdjacencyMatrix(trainingData,index,neighbors)
    #calculate pagerank scores
    PR_Values = {}
    for i,user in enumerate(user_in_validation_set):
        PR_Values[user] = personalizedPageRank(index[user],0.5,AjacencyMatrix)
        if i%100==0:
            print('progress:',i,'/',len(user_in_validation_set))
    scores = {}
    for pair in validationData:
        scores[pair[0]+' '+pair[1]] = PR_Values[pair[0]][index[pair[1]]] + PR_Values[pair[1]][index[pair[0]]]
    top100Links = sorted(scores.items(), key=itemgetter(1))[::-1][:100]
    #validation & evaluation
    groundTruth={}.fromkeys([pair[0]+' '+pair[1] for pair in valid_pos])
    evaluation(top100Links,groundTruth)

if __name__=='__main__':
    main()