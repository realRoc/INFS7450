import networkx as nx
import numpy as np

# Task One
def betweenness_centrality(G):
    betweenness = dict.fromkeys(G, 0.0)
    nodes = G
    for node in nodes:
        # get each node's number of shortest paths by BFS
        S, P, sigma = _bfs_shortest_path(G, node)
        # accumulate the fraction to compute the betweenness
        betweenness = _accumulator(betweenness, S, P, sigma, node)
    return betweenness


def _bfs_shortest_path(G, node):
    '''
    Using BFS to compute the shortest paths from source to target.
    Parameters:
        G: graph
        node: source node
    Returns:
        S (list): list of nodes which have been computed
        P (dictionary): (key) nodes, (value) shortest path from source to key
        sigma (dictionary): (key) nodes, (value) number of shortest paths from source to key
    '''
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)
    D = {}
    sigma[node] = 1.0
    D[node] = 0
    Q = [node]
    while Q:
        v = Q.pop(0)
        S.append(v)
        Dv = D[v]
        sigmav = sigma[v]
        for w in G[v]:
            if w not in D:
                Q.append(w)
                D[w] = Dv + 1
            if D[w] == Dv + 1:
                sigma[w] += sigmav
                P[w].append(v)
    return S, P, sigma


def _accumulator(betweenness, S, P, sigma, s):
    # fraction dictionary
    Cb = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1.0 + Cb[w]) / sigma[w]
        # for each node other than node s
        for v in P[w]:
            Cb[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += Cb[w]
    return betweenness

# generate a net G
G = nx.read_edgelist('C:/Users/simon/Desktop/INFS7450/Project/INFS7450_Project_One/INFS7450-Project_One/3_data.txt', 
nodetype=int)
# compute the betweenness centrality
centrality_dic = betweenness_centrality(G)
# print the top 10 node name and values
print('Top 10 betweenness centrality (node,value): ')
print(sorted(centrality_dic.items(), key=lambda item: item[1], reverse=True)[0:10])
print('\n')

# Task Two
# Ground_Truth = nx.pagerank(G)

ALPHA = 0.85
BETA = 0.15
size = len(G.nodes())
# adjacency matrix A
A = nx.to_numpy_matrix(G)
# inverse of the degree matrix D
degrees = np.array(G.degree(G.nodes()))
D = np.eye(size)/np.delete(degrees, 0, 1)

# matrix method (which is the same as nx.pagerank_numpy)
''' Matrix Method (Same as networkx.pagerank_numpy)
PageRank = BETA*np.linalg.inv(np.eye(size)-ALPHA*A*D)*np.ones((size,1))
PageRank_dic = dict(zip(G.nodes(),PageRank))
print(sorted(PageRank_dic.items(), key=lambda item: item[1], reverse=True)[0:10])
'''

# power iteration method
C = np.ones((size,1))
iter = 10
a = A*D
for i in range(iter):
    C = ALPHA*a*C + BETA
PageRank = dict(zip(G.nodes(),C))
# print the top 10 node name and values
print('Top 10 Page Rank (node,value): ')
print(sorted(PageRank.items(), key=lambda item: item[1], reverse=True)[0:10])
