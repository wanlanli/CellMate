# Copyright 2024 wlli
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import networkx as nx
import matplotlib.pyplot as plt


def node_color(nodes, key: int = 0, p: int = 0, m: int = 0):
    node_color = []
    for n in nodes:
        if n == key:
            c = "blue"
        elif n == m:
            c = "red"
        elif n == p:
            c = "green"
        else:
            c = "lightblue"
        node_color.append(c)
    return node_color


def draw_subgraph(g, key, p=0, m=0):
    f, ax = plt.subplots(figsize=(5, 5))
    undirected_g = g.to_undirected()
    nodes = nx.node_connected_component(undirected_g, key)
    subgraph = nx.subgraph(undirected_g, nodes)
    pos = nx.bfs_layout(subgraph, start=key)
    subgraph_directed = nx.subgraph(g, nodes)
    node_colors = node_color(list(subgraph_directed.nodes), key, p, m)
    nx.draw(subgraph_directed, with_labels=True,
            pos=pos,
            font_weight='bold',
            node_color=node_colors, node_size=500, ax=ax)


def draw_graph_by_layer(network):
    un_network = network.to_undirected()
    groups = list(nx.connected_components(un_network))
    f, ax = plt.subplots(len(groups), 1, figsize=(7, 3*len(groups)))
    for i, g in enumerate(groups):
        subgraph = nx.subgraph(un_network, g)
        pos = nx.bfs_layout(subgraph, start=min(g))
        subgraph_directed = nx.subgraph(network, g)
        nx.draw(subgraph_directed, with_labels=True,
                pos=pos,
                font_weight='bold',
                node_size=500, ax=ax[i])
