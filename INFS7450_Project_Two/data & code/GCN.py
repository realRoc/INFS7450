# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 09:08:57 2020

@author: Yupeng Wu 45960600

Ref: https://github.com/stellargraph/stellargraph
     https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/gcn-link-prediction.html
     
Dear tutor:
    Although this GCN does not work as expected, I want to show my work to you.
    I've sent an email to ask for hints for GCN with no reply yet.
    Please give any feedback about GCN if possible. I'm really curious about how GCN works.
    Thanks in advance.
"""

import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import FullBatchLinkGenerator
from stellargraph.layer import GCN, LinkEmbedding


from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection

from stellargraph import globalvar
from stellargraph import datasets
from IPython.display import display, HTML
#%matplotlib inline

import pandas as pd
import numpy as np

import networkx as nx
from random import choice


G_train_nx = nx.read_edgelist('training.txt')
nodes_train = G_train_nx.nodes()
feature_vector = [1,1,1,1,1]
node_data = pd.DataFrame(
    [feature_vector for i in range(len(nodes_train))],
    index=[node for node in nodes_train])
G_train = sg.StellarGraph.from_networkx(G_train_nx, node_features=node_data)

G_test_nx = nx.read_edgelist('val_positive.txt')
nodes = G_test_nx.nodes()
feature_vector = [1,1,1,1,1]
node_data = pd.DataFrame(
    [feature_vector for i in range(len(nodes))],
    index=[node for node in nodes])
G_test = sg.StellarGraph.from_networkx(G_test_nx, node_features=node_data)

G_test_neg = nx.read_edgelist('val_negative.txt')
edges_test_neg = G_test_neg.edges()

edges_train_neg = []
i = 0
while i < len(nodes_train):
    node = choice(list(nodes_train))
    target = choice(list(nodes_train))
    if node != target and (node, target) not in G_train_nx.edges():
        edges_train_neg.append((node, target))
        i += 1

edge_ids_train = G_train.edges()
edge_labels_train = [1 for i in range(len(G_train.edges()))]
for neg_edge in edges_train_neg:
    edge_ids_train.append(neg_edge)
    edge_labels_train.append(0)
print(G_train.info())

edge_ids_test = G_test.edges()
edge_labels_test = [1 for i in range(len(G_test.edges()))]
for neg_edge in edges_test_neg:
    edge_ids_test.append(neg_edge)
    edge_labels_test.append(0)
print(G_test.info())

epochs = 50

train_gen = sg.mapper.FullBatchLinkGenerator(G_train, method="gcn")
train_flow = train_gen.flow(edge_ids_train, edge_labels_train)

test_gen = FullBatchLinkGenerator(G_test, method="gcn")
test_flow = train_gen.flow(edge_ids_test, edge_labels_test)

gcn = GCN(
    layer_sizes=[16, 16], activations=["relu", "relu"], generator=train_gen, dropout=0.3
)

x_inp, x_out = gcn.in_out_tensors()

prediction = LinkEmbedding(activation="relu", method="ip")(x_out)
prediction = keras.layers.Reshape((-1,))(prediction)

model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=keras.optimizers.Adam(lr=0.01),
    loss=keras.losses.binary_crossentropy,
    metrics=["accuracy"],
)

init_train_metrics = model.evaluate(train_flow)
init_test_metrics = model.evaluate(test_flow)

print("\nTrain Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

history = model.fit(
    train_flow, epochs=epochs, validation_data=test_flow, verbose=1, shuffle=False
)

sg.utils.plot_history(history)

train_metrics = model.evaluate(train_flow)
test_metrics = model.evaluate(test_flow)

print("\nTrain Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))







