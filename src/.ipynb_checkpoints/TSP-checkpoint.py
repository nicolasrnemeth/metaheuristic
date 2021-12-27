class TSP(object):
    # Global variables
    routes = dict()
    node_coordinates = np.nan
    distance_matrix = np.nan
    
    def __init__(self, node_coord, dist_matrix):
        TSP.node_coordinates = node_coord
        TSP.distance_matrix = dist_matrix
        # Random solution to start with
        self.start_path = list(np.random.permutation(len(node_coord)))
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