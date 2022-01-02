# Import required packages 
import numpy as np

class Path(object):
    """ Object to represent a path in the TSP problem """
    def __init__(self, path):
        self.path = path
        self.node_count = len(path)
        self.path_length = TSP.get_length(path)
        self.edges = set()
        for i in range(self.node_count):
            self.edges.add( tuple( sorted( [self.path[i-1], self.path[i]] ) ) )
    
    def __getitem__(self, idx):
        """ Given an index return the node ID """
        return self.path[idx]

    def __contains__(self, edge):
        """ Check if a given connection is present in the path """
        return edge in self.edges

    def get_index(self, node):
        """ Get position of node in path """
        return self.path.index(node)

    def neighbors(self, node):
        """ Get previous and next node of the input node """
        idx = self.path.index(node)
        return (self.path[idx-1], self.path[idx+1 if idx+1 != self.node_count else 0])

    def create_path(self, edges_remove, edges_add):
        """ Given set of edges to remove and add create a new path """
        edges = self.edges - edges_remove
        edges.update(edges_add)
        
        # Check if there are enough edges to create a path
        if len(edges) < self.node_count:
            return False, list()
        
        # We can start from an arbitrary node, in this case 0
        node = 0
        # Create all successors
        next_nodes = dict()
        while edges != set():
            for l, r in edges:
                if l == node:
                    next_nodes[node] = r
                    node = r
                    break
                if r == node:
                    next_nodes[node] = l
                    node = l
                    break
            edges -= set([(l, r)])
            
        # Check if there are enough nodes to create a path
        if len(next_nodes) < self.node_count:
            return False, list()

        new_path = [0]
        _next_ = next_nodes[0]
        seen = set(new_path)

        # Stop if a node has been visited twice
        while _next_ not in seen:
            seen.add(_next_)
            new_path.append(_next_)
            _next_ = next_nodes[_next_]

        # If all nodes have been visited we have a valid path
        isValidPath = len(new_path) == self.node_count
        return isValidPath, new_path

class TSP(object):
    # Static variables 
    node_coordinates = np.nan
    distance_matrix = np.nan
    
    """ Object to hold data for given TSP problem """
    def __init__(self, node_coord, dist_matrix, _rng_=None):
        TSP.node_coordinates = node_coord
        TSP.distance_matrix = dist_matrix
        # Random solution to start with
        # Make results reproducible by setting RNG seed
        RNG = np.random.default_rng(_rng_)
        self.start_path = list(RNG.permutation(len(dist_matrix)))
        self.start_path_length = TSP.get_length(self.start_path)
        self.Path = Path(self.start_path)
        # Current heuristic optimization path
        self.path = self.start_path
        self.path_length = self.start_path_length
    
    @staticmethod
    def get_length(path):
        """ Compute path cost/length """
        length = TSP.distance_matrix[path[0], path[-1]]
        for idx in range(len(path)-1):
            length += TSP.distance_matrix[path[idx], path[idx+1]]
        return length