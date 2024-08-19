# (C) Copyright 2023 Beijing Academy of Quantum Information Sciences
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import networkx as nx
import numpy as np


def wl_subtree_kernel(g1: nx.Graph, g2: nx.Graph, iteration: int = 3):
    """Compute the Weisfeiler-Lehman Subtree Kernel between two graphs.

    References:
        Shervashidze, N., Schweitzer, P., Leeuwen, E.J.v., Mehlhorn,
        K., Borgwardt, K.M., 2011. Weisfeiler-lehman graph kernels.
        Journal of Machine Learning Research 12, 2539–2561.

    Args:
        g1 (nx.Graph):
        g2 (nx.Graph):
        iteration (int): Maximum height for the subtree kernel computation.

    Returns:
        kernel_value: Subtree kernel similarity between g1 and g2.
    """
    kernel_value = 0
    for i in range(iteration):
        kernel_value = kernel_value + _subtree_kernel(g1, g2)
        g1, g2 = _iteration_graph(g1, g2)
    return kernel_value


def _subtree_kernel(g1: nx.Graph, g2: nx.Graph):
    """Compute the Subtree Kernel between two graphs.
    """
    value = 0
    for items1 in g1.edges.data():
        from1 = g1.nodes[items1[0]]['weight']
        to1 = g1.nodes[items1[1]]['weight']
        if from1 > to1:
            mid = from1
            from1 = to1
            to1 = mid
        weight1 = items1[2]['weight']
        for items2 in g2.edges.data():
            from2 = g2.nodes[items2[0]]['weight']
            to2 = g2.nodes[items2[1]]['weight']
            if from2 > to2:
                mid = from2
                from2 = to2
                to2 = mid
            weight2 = items2[2]['weight']
            if from1 == from2 and to1 == to2:
                value = value + 1 / np.exp((weight1 - weight2) ** 2)
    return value


def _iteration_graph(g1: nx.Graph, g2: nx.Graph):
    """Iteratively generate subtree graph.
    """
    num = 0
    dic = {}
    res1 = {}
    res2 = {}
    for items1 in g1.nodes.data():
        weight1 = items1[1]['weight']
        for items2 in g2.nodes.data():
            weight2 = items2[1]['weight']
            if weight1 > num:
                num = weight1
            if weight2 > num:
                num = weight2
    num = num + 1
    for items1 in g1.nodes.data():
        key = str(items1[1]['weight']) + ","
        rellist = []
        neilist = [n for n in g1.neighbors(items1[0])]
        for i in neilist:
            rellist.append(g1.nodes[i]['weight'])
        rellist.sort()
        for i in rellist:
            key = key + str(i)
        if key not in dic.keys():
            dic[key] = num
            res1[items1[0]] = num
            num = num + 1
        else:
            res1[items1[0]] = dic[key]
    for items2 in g2.nodes.data():
        key = str(items2[1]['weight']) + ","
        rellist = []
        neilist = [n for n in g2.neighbors(items2[0])]
        for i in neilist:
            rellist.append(g2.nodes[i]['weight'])
        rellist.sort()
        for i in rellist:
            key = key + str(i)
        if key not in dic.keys():
            dic[key] = num
            res2[items2[0]] = num
            num = num + 1
        else:
            res2[items2[0]] = dic[key]
    for keys in res1.keys():
        g1.add_node(keys, weight=res1[keys])
    for keys in res2.keys():
        g2.add_node(keys, weight=res2[keys])
    return g1, g2


def fast_subtree_kernel(g1, g2, iteration: int = 3):
    """Compute the Fast Subtree Kernel between two graphs.

    References:
        Shervashidze, N., & Borgwardt, K. M. (2009). Fast subtree kernels on graphs.
        In Advances in Neural Information Processing Systems (pp. 1660-1668).

    Args:
        g1 (nx.Graph):
        g2 (nx.Graph):
        iteration (int): Maximum height for the subtree kernel computation.

    Returns:
        Subtree kernel similarity between g1 and g2.
    """
    labels1 = _get_labels(g1)
    labels2 = _get_labels(g2)
    node_pairs1 = _get_node_pairs(g1)
    node_pairs2 = _get_node_pairs(g2)
    subtrees1 = _get_subtrees(g1, node_pairs1)
    subtrees2 = _get_subtrees(g2, node_pairs2)
    print(labels1, labels2)
    print(node_pairs1, node_pairs2)
    K = np.zeros((len(node_pairs1), len(node_pairs2)))
    for i in range(iteration):
        for j in range(len(node_pairs1)):
            for k in range(len(node_pairs2)):
                if i == 0:
                    if labels1[node_pairs1[j][0]] == labels2[node_pairs2[k][0]]:
                        K[j][k] = 1
                else:
                    subtree1 = subtrees1[node_pairs1[j]]
                    subtree2 = subtrees2[node_pairs2[k]]
                    if nx.is_isomorphic(subtree1, subtree2):
                        K[j][k] += K[j - 1][k - 1]
    return K[-1][-1]


def _get_labels(graph):
    """Get labels of graph.
    """
    labels = []
    for node in graph.nodes():
        label = str(graph.nodes[node]['weight'])
        for neighbor in graph.neighbors(node):
            label += '_' + str(graph.nodes[neighbor]['weight'])
        labels.append(label)
    return labels


def _get_node_pairs(graph):
    """Obtain all node pairs in graph.
    """
    node_pairs = []
    for node in graph.nodes():
        for neighbor in graph.neighbors(node):
            node_pairs.append((node, neighbor))
    return node_pairs


def _get_subtrees(graph, node_pairs):
    """Generate a subgraph based on node pairs.
    """
    subtrees = {}
    for node_pair in node_pairs:
        subtree = nx.Graph()
        subtree.add_edges_from(nx.all_shortest_paths(graph, node_pair[0], node_pair[1]))
        subtrees[node_pair] = subtree
    return subtrees
