# Import required packages 
import numpy as np

class Route(object):
    def __init__(self, route):
        self.route = route
        self.city_count = len(route)
        self.route_length = TSP.get_length(route)
        self.create_edges()

    def create_edges(self):
        self.edges = set()
        for i in range(self.city_count):
            edge = tuple(sorted([self.route[i-1], self.route[i]]))
            self.edges.add(edge)
            
    def __getitem__(self, idx):
        return self.route[idx]

    def __contains__(self, edge):
        return edge in self.edges

    def get_index(self, node):
        return self.route.index(node)

    # Get previous and next node of current
    def neighbors(self, node):
        idx = self.route.index(node)
        prev = idx - 1
        next_ = idx + 1
        if next_ == self.city_count:
            next_ = 0

        return (self.route[prev], self.route[next_])

    def prev(self, idx):
        return self.route[idx-1]
    def next_(self, idx):
        return self.route[idx+1]

    def generate(self, edges_remove, edges_add):
        edges = (self.edges - edges_remove) | edges_add
        
        if len(edges) < self.city_count:
            return False, list()
        node = 0
        successors = {}
        # Build the list of successors
        while len(edges) > 0:
            for i, j in edges:
                if i == node:
                    successors[node] = j
                    node = j
                    break
                elif j == node:
                    successors[node] = i
                    node = i
                    break
            edges.remove((i, j))
        # Similarly, if not every node has a successor, this can not work
        if len(successors) < self.city_count:
            return False, list()

        _next_ = successors[0]
        new_route = [0]
        seen = set(new_route)

        # If we already encountered a node it means we have a loop
        while _next_ not in seen:
            seen.add(_next_)
            new_route.append(_next_)
            _next_ = successors[_next_]

        # If we visited all nodes without a loop we have a tour
        return len(new_route) == self.city_count, new_route

class TSP(object):
    # Global variables
    routes = dict()
    node_coordinates = np.nan
    distance_matrix = np.nan
    
    def __init__(self, node_coord, dist_matrix, _rng_=None):
        TSP.node_coordinates = node_coord
        TSP.distance_matrix = dist_matrix
        # Random solution to start with
        # Make results reproducible by setting RNG istance
        RNG = np.random.default_rng(_rng_)
        self.start_path = list(RNG.permutation(len(node_coord)))
        self.start_path_length = TSP.get_length(self.start_path)
        self.Route = Route(self.start_path)
        # Current heuristic optimization path
        self.path = self.start_path
        self.path_length = self.start_path_length
    
    # Compute path length
    @staticmethod
    def get_length(path):
        length = TSP.distance_matrix[path[0], path[-1]]
        for idx in range(len(path)-1):
            length += TSP.distance_matrix[path[idx], path[idx+1]]
        return length
    
    # Remember encountered paths
    def remember(self, path, length):
        self.path = path
        self.path_length = length
        TSP.routes[tuple(sorted(path))] = dict(path=path, length=length)