class UnionFindDisjointSet:
    def __init__(self, n_node):
        self.parent = [i for i in range(n_node)]

    def find(self, node):
        if self.parent[node] == node:
            return node

        self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def is_same_set(self, node_x, node_y):
        return self.find(node_x) == self.find(node_y)

    def join(self, node_x, node_y):
        self.parent[self.find(node_x)] = self.find(node_y)
