import sys
import numpy as np
from os import path
from scipy.spatial.distance import squareform, pdist

class FileParser(object):
    """ Parse TSP input file """
    def __init__(self, filepath, edge_weight_section=False):
        self.filepath = filepath
        self.distMatrix = None
        self.coordinates = None
        self.edge_weight_section = edge_weight_section
    
    # Input given as node coordinates
    def node_coord(self, idx):
        with open(self.filepath, 'r') as ifile:
            nodes = list()
            for line in ifile.readlines()[idx+1:]:
                if line.strip() == "EOF":
                    break
                x, y = line.strip().split(" ")[1:3]
                nodes.append([float(x), float(y)])
            nodes = np.array(nodes)
            self.coordinates = nodes
            self.distMatrix = squareform( pdist( nodes, metric="euclidean" ) )
        
    # Input given as edge weights
    def edge_weight(self, idx):
        weights = list()
        with open(self.filepath, 'r') as ifile:
            for line in ifile.readlines()[idx+1:]:
                if line.strip() == "DISPLAY_DATA_SECTION":
                    break
                weights += line.strip().split(" ")
                
            weights = [float(x) for x in weights]
            n = weights.count(0)
            distMatrix = np.zeros((n, n))
            rowIdx = 0
            colIdx = 0
            for w in weights:
                if w == 0:
                    if self.format == "UPPER_DIAG_ROW":
                        colIdx += 1
                        rowIdx = 0
                        continue
                    rowIdx += 1
                    colIdx = 0
                    continue
                distMatrix[rowIdx, colIdx] = w
                colIdx += 1
            self.distMatrix = distMatrix+distMatrix.T

    def parse(self):
        if not path.exists(self.filepath):
            print("Inputfile could not be found, please check filename and path.")
            sys.exit(0)
            
        with open(self.filepath, 'r') as ifile:
            for idx, line in enumerate(ifile.readlines()):
                if line.strip() == "NODE_COORD_SECTION" or line.strip() == "DISPLAY_DATA_SECTION":
                    self.node_coord(idx)
                    return self
                if self.edge_weight_section:
                    if line.strip().split(" ")[0] == "EDGE_WEIGHT_FORMAT":
                        self.format = line.strip().split(" ")[1]
                    if line.strip() == "EDGE_WEIGHT_SECTION":
                        self.edge_weight(idx)
                        return self
            if line.strip().split(" ")[0].isdigit():
                print("The input file is in the wrong format (see README.md for format specs).")
                sys.exit(0)
        return self
    
    # Return parsed data
    def parsed_data(self):
        return self.distMatrix, self.coordinates
    

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
        edges.add(edges_add)
        
        # Check if there are enough edges to create a path
        if len(edges) < self.node_count:
            return False, list()
        
        # We can start from an arbitrary node, in this case 0
        node = 0
        # Track the next node of each one in a dict
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
            edges -= (l, r)
            
        # Check if each node has a degree of 2
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
        self.start_path = list(RNG.permutation(len(node_coord)))
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
        
        
class LinKernighan(object):
    """ Heuristic to solve TSP """
    def __init__(self, Tsp, doAnalysis=False):
        self.Tsp = Tsp
        self.solutions = set()
        self.doAnalysis = True if doAnalysis else False
        # Create a potential neighbor list for each node
        self.neighbors  = dict()
        for node in self.Tsp.path:
            self.neighbors[node] = list()
            self.neighbors[node] += [n for n in range(len(TSP.distance_matrix)) if n != node]
            
    def restart(self):
        """ Part of algorithm which is restarted when better path was found """
        path = Path(self.Tsp.path)
        # Perform all 2-opt moves
        for n1 in self.Tsp.path:
            neighbors = path.neighbors(n1)    
            for n2 in neighbors:
                edges_remove = set([tuple(sorted([n1, n2]))])
                profit_gain = TSP.distance_matrix[n1, n2]
                # Get nearest neighbors based on potential gain
                near = self.nearest_neighbors(n2, path, profit_gain, edges_remove, set())
               
                # Track number of searches
                num_search = 0
                for n3, (gain_diff, gain_i) in near:
                    # Check if new node does not belong to the current path
                    if n3 in neighbors:
                        continue    
                    edges_add = set([tuple(sorted([t2, t3]))])
                    
                    # Restart the algorithm after a better path was found
                    if self.chooseX(route, t1, t3, gain_i, edges_remove, edges_add):
                        return True
                    
                    num_search += 1
                    # Set number of searches to perform before to try a new node n2
                    if num_search == 5:
                        break
        return False
    
    def optimize(self):
        """ Start heuristic optimization """
        while True:
            # Keep track of improved solutions to check for
            # duplicate solutions during optimization run
            self.solutions.add(tuple(self.Tsp.path))
            if not self.restart():
                break
        return self.Tsp.path, self.Tsp.path_length
    
    def nearest_neighbors(self, node_relink, path, profit_gain, edges_remove, edges_add):
        """ Find nearest neighbors of a node based on the potential gain """
        neighbors_gain = dict()
        
        for node in self.neighbors[node_relink]:
            edge_add_i = tuple(sorted([node_relink, node]))
            gain_i = profit_gain - TSP.distance_matrix[node_relink, node]
            # SKIP if edge to add
            # 1) has no profit gain
            # 2) or is present in edges to remove
            # 3) or is present in the current path
            if gain_i <= 0 or edge_add_i in edges_remove or edge_add_i in path:
                continue

            for _next_ in self.neighbors[node]:
                edge_remove_i = tuple(sorted([node, _next_]))
                
                # Check if edge to remove is not in set of edges to remove or add
                if edge_remove_i not in edges_remove and edge_remove_i not in edges_add:
                    gain_diff = TSP.distance_matrix[node, _next_] - TSP.distance_matrix[node_relink, node]

                    if node in neighbors_gain and gain_diff > neighbors_gain[node][0]:
                        neighbors_gain[node] = [gain_diff, None]
                    else:
                        neighbors_gain[node] = [gain_diff, gain_i]
                        
        neighbors_gain = sorted(neighbors_gain.items(), key=lambda x: x[1][0], reverse=True)
        return neighbors_gain

    def select_edge_remove(self, path, n1, last, gain, edges_remove, edges_add):
        """ Select which edge to remove from the current path """
        if len(edges_remove) == 4:
            prev, next_ = path.neighbors(last)

            # Remove more promising edge (i.e. edge which higher cost)
            if TSP.distance_matrix[prev, last] > TSP.distance_matrix[next_, last]:
                neighbor = [prev]
            else:
                neighbor = [next_]
        else:
            neighbor = path.neighbors(last)

        for node_relink in neighbor:
            edge_remove = tuple(sorted([last, node_relink]))
            # Current gain
            gain_i = gain + TSP.distance_matrix[last, node_relink]

            # Check if set of edges to remove and add are disjoint
            if edge_remove not in edges_add and edge_remove not in edges_remove:
                added = edges_add.copy()
                removed = edges_remove.copy()

                removed.add(edge_remove)
                # Relink the path 
                added.add(tuple(sorted([node_relink, n1])))

                relink_profit = gain_i - TSP.distance_matrix[node_relink, n1]
                is_path, new_path = path.create_path(removed, added)

                # Skip if current solution is not a valid path
                if not is_path and len(added) > 2:
                    continue

                # Stop the search when we arrive at a duplicate solution
                if str(new_path) in self.solutions:
                    return False

                # Save the path if it is an improvement
                if is_path and relink_profit > 0:
                    self.Tsp.path = new_path
                    self.Tsp.path_length -= relink_profit
                    return True
                else:
                    # Forward removed edge but select another edge to add
                    selection_add = self.select_edge_add(route, n1, node_relink, gain_i, removed, edges_add)

                    if len(edges_remove) == 2 and selection_add:
                        return True
                    else:
                        return selection_add
        return False

    def select_edge_add(self, path, t1, tail, gain_i, edges_remove, edges_add):
        """ Select which edge to add to the current path """
        near = self.nearest_neighbors(tail, route, gain_i, edges_remove, edges_add)
        # Choose maximum number of searches to perform for 2-opt
        num_search = 5 if len(edges_remove) == 2 else 1
        
        for node, (gain_diff, gain_i) in near:
            edge_add = tuple(sorted([tail, node]))
            added = edges_add.copy()
            added.add(edge_add)

            # Stop search when better path is found
            if self.select_edge_remove(path, t1, node, gain_i, edges_remove, added):
                return True

            num_search -= 1
            # Stop when maximum number of searches reaches
            if num_search == 0:
                return False
            
        return False
    

# Parse input data
fileparser = FileParser("../input_data/berlin52.txt").parse()
distance_matrix, coordinates = fileparser.parsed_data()

# Instantiations
Tsp = TSP(coordinates, distance_matrix, _rng_=None)
LinKer = LinKernighan(Tsp, False)
# Perform heuristic optimization
approx_solution, path_length = LinKer.optimize()

print("")
print(path_length)
print("")
print(approx_solution)