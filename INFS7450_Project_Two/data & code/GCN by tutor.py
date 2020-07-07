#coding:utf8

from random import choice
import tensorflow as tf
import numpy as np
from math import sqrt
from collections import defaultdict

class GCN(object):
    def __init__(self,trainingSet,index,dimension,epoch,lRate,batchSize):
        self.trainingSet = trainingSet
        self.index = index
        self.dimension = dimension
        self.epoch = epoch
        self.lRate = lRate
        self.batchSize = batchSize
        self.dataSize = len(self.trainingSet)
        self.pairs = defaultdict(dict)
        for pair in self.trainingSet:
            self.pairs[pair[0]][pair[1]] = 1
            self.pairs[pair[1]][pair[0]] = 1

    def next_batch(self):
        batch_id = 0
        while batch_id < self.dataSize:
            if batch_id + self.batchSize <= self.dataSize:
                postivePairs = [self.trainingSet[idx] for idx in range(batch_id, self.batchSize + batch_id)]
                batch_id += self.batchSize
            else:
                postivePairs = [self.trainingSet[idx] for idx in range(batch_id, self.dataSize)]
                batch_id = self.dataSize

            a1_idx, a2_idx, neg_idx = [], [], []
            authors = self.index.keys()

            for i, pair in enumerate(postivePairs):
                a1_idx.append(self.index[pair[0]])
                a2_idx.append(self.index[pair[1]])
                #sampling negative examples
                negative_node = choice(authors)
                while negative_node in self.pairs[pair[0]] and negative_node in self.pairs[pair[1]]:
                    negative_node = choice(authors)
                neg_idx.append(self.index[negative_node])

            yield a1_idx, a2_idx, neg_idx

    def initModel(self):
        #construct normalized sparse adjacency matrix
        indices = [[self.index[pair[0]],self.index[pair[1]]] for pair in self.trainingSet]
        values = [1.0/sqrt(len(self.pairs[pair[0]]))/sqrt(len(self.pairs[pair[1]])) for pair in self.trainingSet]
        norm_adj = tf.SparseTensor(indices=indices, values=values, dense_shape=[len(self.index),len(self.index)])
        #initialize embeddings and network parameters
        self.node_embeddings = tf.Variable(tf.truncated_normal(shape=[len(self.index), self.dimension], stddev=0.005),name='Nodes')
        self.isTraining = tf.placeholder(tf.int32)
        self.isTraining = tf.cast(self.isTraining, tf.bool)
        self.weights = dict()
        ego_embeddings = self.node_embeddings
        initializer = tf.contrib.layers.xavier_initializer()
        weight_size = [self.dimension,self.dimension,self.dimension] #can be changed
        weight_size_list = [self.dimension] + weight_size

        self.n_layers = 3 #depth of GCN
        for k in range(self.n_layers):
            self.weights['W_%d_1' % k] = tf.Variable(
                initializer([weight_size_list[k], weight_size_list[k + 1]]), name='W_%d_1' % k)
            self.weights['W_%d_2' % k] = tf.Variable(
                initializer([weight_size_list[k], weight_size_list[k + 1]]), name='W_%d_2' % k)

        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            side_embeddings = tf.sparse_tensor_dense_matmul(norm_adj, ego_embeddings)
            sum_embeddings = tf.matmul(side_embeddings + ego_embeddings, self.weights['W_%d_1' % k])
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            bi_embeddings = tf.matmul(bi_embeddings, self.weights['W_%d_2' % k])

            ego_embeddings = tf.nn.leaky_relu(sum_embeddings + bi_embeddings)

            # message dropout.
            def without_dropout():
                return ego_embeddings
            def dropout():
                return tf.nn.dropout(ego_embeddings, keep_prob=0.95)

            ego_embeddings = tf.cond(self.isTraining,lambda:dropout(),lambda:without_dropout())

            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        self.a1_idx = tf.placeholder(tf.int32, name="a1_idx")
        self.a2_idx = tf.placeholder(tf.int32, name="a2_idx")
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.a1_embedding = tf.nn.embedding_lookup(all_embeddings, self.a1_idx)
        self.a2_embedding = tf.nn.embedding_lookup(all_embeddings, self.a2_idx)
        self.neg_embedding = tf.nn.embedding_lookup(all_embeddings, self.neg_idx)
        self.predict = tf.reduce_sum(tf.multiply(self.a1_embedding,self.a2_embedding),1)

    def buildModel(self):

        # construct loss function add regularization term
        y1 = tf.reduce_sum(tf.multiply(self.a1_embedding, self.a2_embedding), 1)
        y2 = tf.reduce_sum(tf.multiply(self.a1_embedding, self.neg_embedding), 1)
        y3 = tf.reduce_sum(tf.multiply(self.a2_embedding, self.neg_embedding), 1)
        reg = 0.0001 * (tf.nn.l2_loss(self.a1_embedding) + tf.nn.l2_loss(self.a2_embedding) + tf.nn.l2_loss(self.neg_embedding))
        cross_entropy_loss = -tf.reduce_sum(tf.log(tf.sigmoid(y1)) + tf.log(1-tf.sigmoid(y2))) + reg
        pairwise_loss = -tf.reduce_sum(tf.log(tf.sigmoid(y1-y2-y3))) + reg

        loss = pairwise_loss

        #optimization
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        for iteration in range(self.epoch):
            for n, batch in enumerate(self.next_batch()):
                a1_idx, a2_idx, neg_idx = batch
                _, l = self.sess.run([train, loss],
                                feed_dict={self.a1_idx: a1_idx, self.a2_idx: a2_idx, self.neg_idx: neg_idx,self.isTraining:1})
                print('training:', iteration + 1, 'batch', n, 'loss:', l)

    def predictScore(self, a1,a2):
        return self.sess.run(self.predict,feed_dict={self.a1_idx:a1,self.a2_idx:a2,self.isTraining:0})


def readData(filename):
    data = []
    with open(filename) as f:
        for line in f:
         data.append(line.strip().split())
    return data

def buildIndex(data):
    index = {}
    for pair in data:
        if pair[0] not in index:
            index[pair[0]] = len(index)
        if pair[1] not in index:
            index[pair[1]] = len(index)
    return index

def sort(linkScores):
    sortedList = sorted(linkScores.iteritems(), key=lambda d: d[1])
    sortedList.reverse()
    return sortedList

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
    validationData = valid_pos+valid_neg
    index = buildIndex(trainingData)
    #initialize GCN
    gcn = GCN(trainingData,index,50,300,0.002,2000)
    gcn.initModel()
    #train GCN
    gcn.buildModel()
    #predict
    print('predicting...')
    scores = {}
    for pair in validationData:
        a1 = index[pair[0]]
        a2 = index[pair[1]]
        scores[pair[0]+' '+pair[1]] = gcn.predictScore([a1],[a2])
    top100Links = sort(scores)[:100]
    #validation & evaluation
    groundTruth={}.fromkeys([pair[0]+' '+pair[1] for pair in valid_pos])
    evaluation(top100Links,groundTruth)

if __name__=='__main__':
    main()
