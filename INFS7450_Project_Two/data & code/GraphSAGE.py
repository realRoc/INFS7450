# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 23:51:23 2020

@author: Yupeng Wu 45960600

Ref: https://github.com/stellargraph/stellargraph
     https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/gcn-link-prediction.html
"""

import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, HinSAGE, link_classification

from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection

from stellargraph import globalvar
from stellargraph import datasets
from IPython.display import display, HTML
#%matplotlib 

import pandas as pd
import numpy as np

import networkx as nx
from random import choice


# Generate train graph
G_train_nx = nx.read_edgelist('training.txt')

nodes_train = G_train_nx.nodes()
feature_vector = [1,1,1,1,1]
node_data_train = pd.DataFrame(
    [feature_vector for i in range(len(nodes_train))],
    index=[node for node in nodes_train])
G_train = sg.StellarGraph.from_networkx(G_train_nx, node_features=node_data_train)
# Generate test graph
G_test_nx = nx.read_edgelist('val_positive.txt')

G_test_neg = nx.read_edgelist('val_negative.txt')
edges_test_neg = G_test_neg.edges()

G_test_nx.add_edges_from(edges_test_neg)

node_data_test = pd.DataFrame(
    [feature_vector for i in range(len(G_test_nx.nodes()))],
    index=[node for node in G_test_nx.nodes()])
G_test = sg.StellarGraph.from_networkx(G_test_nx, node_features=node_data_test)
# Generate negative data in train
edges_train_neg = []
i = 0
while i < len(nodes_train):
    node = choice(list(nodes_train))
    target = choice(list(nodes_train))
    if node != target and (node, target) not in G_train_nx.edges():
        edges_train_neg.append((node, target))
        i += 1
# Label train data
edge_ids_train = G_train.edges()
edge_labels_train = [1 for i in range(len(G_train.edges()))]
for neg_edge in edges_train_neg:
    edge_ids_train.append(neg_edge)
    edge_labels_train.append(0)
print(G_train.info())
# Label test data
edge_ids_test = G_test.edges()
edge_labels_test = [1 for i in range(len(G_test.edges()))]
for neg_edge in edges_test_neg:
    edge_ids_test.append(neg_edge)
    edge_labels_test.append(0)
print(G_test.info())


batch_size = 20
epochs = 5

num_samples = [20, 10]

train_gen = GraphSAGELinkGenerator(G_train, batch_size, num_samples)
train_flow = train_gen.flow(edge_ids_train, edge_labels_train, shuffle=True)

test_gen = GraphSAGELinkGenerator(G_test, batch_size, num_samples)
test_flow = test_gen.flow(edge_ids_test, edge_labels_test)

layer_sizes = [2, 20]
graphsage = GraphSAGE(
    layer_sizes=layer_sizes, generator=train_gen, bias=True, dropout=0.3
)

# Build the model and expose input and output sockets of graphsage model
# for link prediction
x_inp, x_out = graphsage.in_out_tensors()

prediction = link_classification(
    output_dim=1, output_act="relu", edge_embedding_method="ip"
)(x_out)

model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-3),
    loss=keras.losses.binary_crossentropy,
    metrics=["acc"],
)

init_train_metrics = model.evaluate(train_flow)
init_test_metrics = model.evaluate(test_flow)

print("\nTrain Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_test_metrics):
    print("\t{}: {:0.4f}".format(name, val))
    
history = model.fit(train_flow, epochs=epochs, validation_data=test_flow, verbose=2)

sg.utils.plot_history(history)

train_metrics = model.evaluate(train_flow)
test_metrics = model.evaluate(test_flow)

print("\nTrain Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))






