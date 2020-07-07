# -*- coding: utf-8 -*-
"""
Created on Thu May 28 20:07:23 2020

@author: simon
"""
import networkx as nx

class B_Centrality:
    """A player in the game"""
    _type = 3

    def __init__(self):
        trainingData = self.readData('training.txt')
        G = self.buildGraph(trainingData)
        self._betweeness_centrality = nx.betweenness_centrality(G)
        
    def readData(self, filename):
        data = []
        with open(filename) as f:
            for line in f:
                data.append(line.strip().split())
        return data
    
    def buildGraph(self, data):
        G = nx.Graph()
        for pair in data:
            G.add_edge(pair[0],pair[1])
        return G
    
    def get_b_centrality(self):
        return self._betweeness_centrality