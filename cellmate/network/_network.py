import networkx as nx


class NetCell(nx.DiGraph):
    """
    DiGraph to represent the generation relationships between objects.
    Edge Information:
    - One point -> Two points: Division
    - Two points -> One point: Fusion
    """
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

    def upstream(self, node):
        """
        Get a given node's ancestors from the DiGraph.

        Parameters:
        -----------
        node: int
            The ID of the node.

        Returns:
        -----------
        ancestors: list
            The list of upstream nodes.
        """
        return [n for n in nx.traversal.bfs_tree(self, node, reverse=True) if n != node]

    def downstream(self, node):
        """
        Get a given node's descendants from the DiGraph.

        Parameters:
        -----------
        node: int
            The ID of the node.

        Returns:
        -----------
        descendants: list
            The list of downstream nodes.
        """
        return [n for n in nx.traversal.dfs_tree(self, node) if n != node]

    def layer(self, node):
        """
        Calculate and return the depth (generation number) of a node in the DiGraph.

        Parameters:
        -----------
        node : int
            The ID of the node.

        Returns:
        ----------
        depth : int
            The depth of the given node in the tree.
            (Includes the number of division and fusion events if applicable.)
        """
        upstream = self.upstream(node)
        if len(self.parentes(node)) > 0:
            return len(upstream)
        else:
            return len(upstream)

    def ancient(self, node):
        """
        Return the last ancestor node resulting from division in the directed graph.

        Parameters:
        -----------
        node : int
            The ID of the node.

        Returns:
        ----------
        ancient_id : int or None
            The ID of the last ancestor node, if it exists; otherwise, None.
        """
        in_edges = self.in_edges(node)
        if len(in_edges) == 1:
            return list(in_edges)[0][0]
        else:
            return None

    def sister(self, node):
        """
        Return the sister node resulting from division in the directed graph.

        Parameters:
        -----------
        node : int
            The ID of the node.

        Returns:
        ----------
        sister_id : int or None
            The ID of the sister node, if it exists; otherwise, None.
        """
        ancient = self.ancient(node)
        if ancient is not None:
            return [t for _, t in self.out_edges(ancient) if t != node][0]
        return None

    def parentes(self, node):
        """
        Return the parentes node resulting from fusion in the directed graph.

        Parameters:
        -----------
        node : int
            The ID of the node.

        Returns:
        ----------
        parentes_id : list or []
            The ID of the parentes node, if it exists; otherwise, None.
        """
        in_edges = self.in_edges(node)
        if len(in_edges) == 2:
            return [s for s, _ in in_edges]
        else:
            return []

    def daughter(self, node):
        """
        Return the daughter nodes resulting from division in the directed graph.

        Parameters:
        -----------
        node : int
            The ID of the node.

        Returns:
        ----------
        daughter_id : list or []
            The ID of the daughter nodes, if it exists; otherwise, None.
        """
        out_edges = self.out_edges(node)
        if len(out_edges) == 2:
            return [t for _, t in out_edges]
        else:
            return []

    def spouse(self, node):
        """
        Return the spouse node resulting from fusion in the directed graph.

        Parameters:
        -----------
        node : int
            The ID of the node.

        Returns:
        ----------
        spouse_id : int or None
            The ID of the spouse node, if it exists; otherwise, None.
        """
        daughter_sex = self.daughter_sex(node)
        if daughter_sex is not None:
            return [s for s, _ in self.in_edges(daughter_sex) if s != node][0]
        else:
            return None

    def daughter_sex(self, node):
        """
        Return the daughter node resulting from fusion in the directed graph.

        Parameters:
        -----------
        node : int
            The ID of the node.

        Returns:
        ----------
        daughter_id : int or None
            The ID of the daughter node, if it exists; otherwise, None.
        """
        out_edges = self.out_edges(node)
        if len(out_edges) == 1:
            return list(out_edges)[0][1]
        else:
            return None

    def generation(self, node):
        upstream = self.upstream(node)
        gen = len(upstream)+1
        return gen

    def feature(self, node):
        """
        Merge the following 7 features into a list:
        [GENERATION, ANCIENT_VG, SISTER, PARENTS, DAUGHTER_VG, SPOUSE, DAUGHTER_SEX]

        Parameters:
        -----------
        node : int
            The ID of the node.

        Returns:
        ----------
        output : list
            A list containing the values of the specified features for the given node.
        """
        output = [0, None, None, [], [], None, None]
        # upstream = self.upstream(node)
        output[0] = self.generation(node)  # len(upstream)+1
        output[1] = self.ancient(node)
        output[2] = self.sister(node)
        output[3] = self.parentes(node)
        output[4] = self.daughter(node)
        output[5] = self.spouse(node)
        output[6] = self.daughter_sex(node)

        # in_edges = self.in_edges(node)
        # if len(in_edges) == 1:
        #     output[1] = list(in_edges)[0][0]
        #     output[2] = [t for _, t in self.out_edges(output[1]) if t != node][0]
        # elif len(in_edges) == 2:
        #     output[3] = [s for s, _ in in_edges]
        #     output[0] += -1

        # out_edges = self.out_edges(node)
        # if len(out_edges) == 2:
        #     output[4] = [t for _, t in out_edges]
        # elif len(out_edges) == 1:
        #     output[6] = list(out_edges)[0][1]
        #     output[5] = [s for s, t in self.in_edges(output[6]) if t != node][0]
        return output

    def relative_position(self):
        # pos = {}
        # for node in self .nodes:
        #     f, x, y = np.where(traced_image%1000==node)
        #     if len(f)>0:
        #         f = np.median(f)
        #         x = np.median(x)
        #         y = np.median(y)
        #         pos[node] = [y, -x]
        pass
