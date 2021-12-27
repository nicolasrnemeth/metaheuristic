import numpy as np
from copy import deepcopy
import os
import sys
import numpy as np
from scipy.spatial.distance import squareform, pdist

class FileParser(object):
    """ Parse TSP input file """
    def __init__(self, filepath, edge_weight_section=False):
        self.filepath = filepath
        self.distMatrix = None
        self.edge_weight_section = edge_weight_section
        self.coordinates = None
    
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
            self.distMatrix = squareform( pdist(nodes, metric="euclidean") )
        
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
                    rowIdx += 1
                    colIdx = 0
                    continue
                distMatrix[rowIdx, colIdx] = w
                colIdx += 1
            self.distMatrix = distMatrix+distMatrix.T

    def parse(self):
        with open(self.filepath, 'r') as ifile:
            for idx, line in enumerate(ifile.readlines()):
                if line.strip() == "NODE_COORD_SECTION" or line.strip() == "DISPLAY_DATA_SECTION":
                    self.node_coord(idx)
                    return self
                if self.edge_weight_section:
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
        if len(successors) < self.size:
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


class LinKernighanHelsgaun(object):
    def __init__(self, Tsp):
        self.solutions = set()
        self.neighbors  = dict()
        self.Tsp = Tsp
    
    def optimize(self):
        # Check if route exists before optimization 
        route = tuple(sorted(self.Tsp.path))
        if route in TSP.routes:
            self.Tsp.path = TSP.routes[route]["path"]
            self.Tsp.path_length = TSP.routes[route]["length"]
            return self.Tsp.path, self.Tsp.path_length
    
        for i in self.Tsp.path:
            self.neighbors[i] = list()

            for j, dist in enumerate(TSP.distance_matrix[i]):
                if dist > 0 and j in self.Tsp.path:
                    self.neighbors[i].append(j)

        # Restart the loop each time we find an improving candidate
        better = True
        while better:
            better = self.improve()
            # Paths always begin at 0 so this should manage to find duplicate solutions
            self.solutions.add(tuple(self.Tsp.path))

        self.Tsp.remember(self.Tsp.path, self.Tsp.path_length)
        
        return self.Tsp.path, self.Tsp.path_length

    def closest(self, node_relink, route, profit_gain, edges_remove, edges_add):
        """
        Find the closest neighbours of a node ordered by potential gain.  As a
        side-effect, also compute the partial improvement of joining a node.

        Parameters:

            - t2i: node to relink from

            - tour: current tour to optimise

            - gain: current gain

            - broken: set of edges to remove (X)

            - joined: set of edges to join (Y)

        Return: sorted list of neighbours based on potential improvement with
        next omission
        """
        neighbors = dict()

        # Create the neighbors of node_relink
        for node in self.neighbors[node_relink]:
            edge_add_i = tuple(sorted([node_relink, node]))
            gain_i = profit_gain - TSP.distance_matrix[node_relink, node]
            
            # Check if edge to add has profit gain, is not present in edges to remove
            # and is not present in the route
            if gain_i <= 0 or edge_add_i in edges_remove or edge_add_i in route:
                continue

            for _next_ in self.neighbors[node]:
                edge_remove_i = tuple(sorted([node, _next_]))

                # TODO verify it is enough, but we do check if the tour is
                # valid first thing in `chooseX` so this should be sufficient
                #
                # Check that "x_i+1 exists"
                if edge_remove_i not in edges_remove and edge_remove_i not in edges_add:
                    gain_diff = TSP.distance_matrix[node, _next_] - TSP.distance_matrix[node_relink, node]

                    if node in neighbors and gain_diff > neighbors[node][0]:
                        neighbors[node][0] = gain_diff
                    else:
                        neighbors[node] = [gain_diff, gain_i]
                        
        neighbors = sorted(neighbors.items(), key=lambda x: x[1][0], reverse=True)
        return neighbors

    def improve(self):
        """
        Start the LKH algorithm with the current tour.
        """
        route = Route(self.Tsp.path)

        # Find all valid 2-opt moves and try them
        for t1 in self.Tsp.path:
            neighbors = route.neighbors(t1)

            for t2 in neighbors:
                edges_remove = set([tuple(sorted([t1, t2]))])
                # Initial savings
                profit_gain = TSP.distance_matrix[t1, t2]

                close = self.closest(t2, route, profit_gain, edges_remove, set())

                # Number of neighbours to try
                num_neighbors = 5

                for t3, (_, gain_i) in close:
                    # Make sure that the new node is none of t_1's neighbours
                    # so it does not belong to the tour.
                    if t3 in neighbors:
                        continue

                    edges_add = set([tuple(sorted([t2, t3]))])

                    if self.chooseX(route, t1, t3, gain_i, edges_remove, edges_add):
                        # Return to Step 2, that is the initial loop
                        return True
                    # Else try the other options

                    num_neighbors -= 1
                    # Explored enough nodes, change t_2
                    if num_neighbors == 0:
                        break

        return False

    def chooseX(self, route, t1, last, gain, edges_remove, edges_add):
        """
        Choose an edge to omit from the tour.

        Parameters:

            - tour: current tour to optimise

            - t1: starting node for the current k-opt

            - last: tail of the last edge added (t_2i-1)

            - gain: current gain (Gi)

            - broken: potential edges to remove (X)

            - joined: potential edges to add (Y)

        Return: whether we found an improved tour
        """
        if len(edges_remove) == 4:
            prev, next_ = route.neighors(last)

            # Give priority to the longest edge for x_4
            if TSP.distance_matrix[prev, last] > TSP.distance_matrix[next_, last]:
                neighbor = [prev]
            else:
                neighbor = [next_]
        else:
            neighbor = route.neighbors(last)

        for node_relink in neighbor:
            edge_remove = tuple(sorted([last, node_relink]))
            # Gain at current iteration
            gain_i = gain + TSP.distance_matrix[last, node_relink]

            # Verify that X and Y are disjoint, though I also need to check
            # that we are not including an x_i again for some reason.
            if edge_remove not in edges_add and edge_remove not in edges_remove:
                added = deepcopy(edges_add)
                removed = deepcopy(edges_remove)

                removed.add(edge_remove)
                added.add(tuple(sorted([node_relink, t1])))  # Try to relink the tour

                relink_cost = gain_i - TSP.distance_matrix[node_relink, t1]
                is_route, new_route = route.generate(removed, added)

                # The current solution does not form a valid tour
                if not is_route and len(added) > 2:
                    continue

                # Stop the search if we come back to the same solution
                if str(new_route) in self.solutions:
                    return False

                # Save the current solution if the tour is better, we need
                # `is_tour` again in the case where we have a non-sequential
                # exchange with i = 2
                if is_route and relink_cost > 0:
                    self.Tsp.path = new_route
                    self.Tsp.path_length -= relink_cost
                    return True
                else:
                    # Pass on the newly "removed" edge but not the relink
                    choice = self.chooseY(route, t1, t2i, gain_i, removed, joined)

                    if len(edges_remove) == 2 and choice:
                        return True
                    else:
                        # Single iteration for i > 2
                        return choice

        return False

    def chooseY(self, route, t1, tail, gain_i, edges_remove, edges_add):
        """
        Choose an edge to add to the new tour.

        Parameters:

            - tour: current tour to optimise

            - t1: starting node for the current k-opt

            - t2i: tail of the last edge removed (t_2i)

            - gain: current gain (Gi)

            - broken: potential edges to remove (X)

            - joined: potential edges to add (Y)

        Return: whether we found an improved tour
        """
        ordered = self.closest(tail, route, gain_i, edges_remove, edges_add)

        if len(edges_remove) == 2:
            # Check the five nearest neighbours when i = 2
            top = 5
        else:
            # Otherwise the closest only
            top = 1

        for node, (_, gain_i) in ordered:
            edge_add = tuple(sorted([tail, node]))
            added = deepcopy(edge_add)
            added.add(edge_add)

            # Stop at the first improving tour
            if self.chooseX(route, t1, node, gain_i, edges_remove, added):
                return True

            top -= 1
            # Tried enough options
            if top == 0:
                return False

        return False


fileparser = FileParser("C:\\Users\\nicol\\Desktop\\co_ws21\\metaheuristic\\input_data\\berlin52.txt").parse()
distance_matrix, coordinates = fileparser.parsed_data()


Tsp = TSP(coordinates, distance_matrix)
LinKerHel = LinKernighanHelsgaun(Tsp)
LinKerHel.optimize()