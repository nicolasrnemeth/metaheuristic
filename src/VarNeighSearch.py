# Import required packages
import time
import numpy as np
from copy import deepcopy
from numpy import inf as Infinity
from TSP import TSP, Path

# Custom class
class VarNeighSearch(object):
    """ Solve TSP with Variable Neighborhood Search """
    def __init__(self, Tsp, max_nodes_2opt=5, max_nodes_2p_opt=1, doAnalysis=False, timeLimit=Infinity, maxR=Infinity):
        self.Tsp = Tsp
        # Maximum size of an r-opt move (default is inifinity)
        # The search space is naturally limited by trying most promising 'x' nodes instead of all
        self.maxR = maxR
        self.timeLimit = timeLimit*60
        self.solutions = set()
        self.doAnalysis = True if doAnalysis else False
        if doAnalysis:
            # Evaluate average path length if 1000 paths are randomly tried
            avgCost_1000randTrials = [TSP.get_length(np.random.permutation(self.Tsp.Path.node_count)) for i in range(1000)]
            avgCost_1000randTrials = np.mean(avgCost_1000randTrials)
            """ Dictionary with collection of analysis data:
            number of nodes, path lengths of each improvement, elapsed time between each improvement,
            number of edges removed/added, i.e. number of opt-moves performed for path improvement,
            execution time of optimization run, x and y coordinates of nodes
            """
            self.analysisData = dict(node_count=deepcopy(self.Tsp.Path.node_count), 
                                     path_lengths=[deepcopy(self.Tsp.start_path_length)], 
                                     elapsed_times=list(), num_moves=list(), exec_time=None, 
                                     avgCost1000Trials=avgCost_1000randTrials, x_coor=list(TSP.node_coordinates[:,0]),
                                     y_coor=list(TSP.node_coordinates[:,1]))
        """ Specify maximum number of nodes to try before to switch to the other 
        neighbor of n2, i.e. before selecting the other edge to remove given starting node n1 """
        self.max_nodes_2opt = max_nodes_2opt
        """ Specify maximum number of nodes to try given that we 
        perform k-opt heuristic where k is larger 2 
        (NB: 2p stands for 2plus meaning larger than 2) """
        self.max_nodes_2p_opt = max_nodes_2p_opt
        # Create a potential neighbor list for each node
        self.neighbors  = dict()
        for node in self.Tsp.path:
            self.neighbors[node] = [n for n in range(len(TSP.distance_matrix)) if n != node]
    
    def optimize(self):
        """ Start heuristic optimization """
        self.start_time = time.time()
        if self.doAnalysis:
            self.elapsed_time = time.time()
        while (time.time() - self.start_time) < self.timeLimit:
            # Keep track of improved solutions to check for
            # duplicate solutions during optimization run
            self.solutions.add(tuple(self.Tsp.path))
            if not self.r_opt():
                break
        if self.doAnalysis:
            self.analysisData["exec_time"] = time.time() - self.start_time
        return self.Tsp.path, self.Tsp.path_length, self.analysisData if self.doAnalysis else None
    
    def r_opt(self):
        """ Perform r-opt improvement heuristic to find better solution. 
        The algorithm starts with r = 2 and increases r as long as not better
        path was encountered so far. """
        path = Path(self.Tsp.path)
        
        for n1 in self.Tsp.path:
            neighbors = path.neighbors(n1)    
            for n2 in neighbors:
                edges_remove = set([tuple(sorted([n1, n2]))])
                # Get nearest neighbors based on potential gain
                near = self.nearest_neighbors(n2, path, TSP.distance_matrix[n1, n2], edges_remove, set())               
                
                # Track number of nodes tried
                num_nodes = 0
                for n3, gain_i in near:
                    # Check if new node does not belong to the current path
                    if n3 in neighbors:
                        continue    
                    
                    # Restart the algorithm after a better path was found
                    if self.select_edge_remove(path, n1, n3, gain_i, edges_remove, set([tuple(sorted([n2, n3]))])):
                        return True
                    else:
                        if (time.time() - self.start_time) > self.timeLimit:
                            return False
                    
                    num_nodes += 1
                    # Maximum number of nodes before to try before to try the other neighbor of n1 -> n2
                    if num_nodes >= self.max_nodes_2opt:
                        break
        return False
    
    def nearest_neighbors(self, node_relink, path, cost, edges_remove, edges_add):
        """ Find nearest neighbors of a node based on the potential gain.
        Try most promising x neighbors instead of all of them or a random subset. """
        neighbors_gain = dict()
        for node in self.neighbors[node_relink]:
            edge_add_i = tuple(sorted([node_relink, node]))
            gain_i = cost - TSP.distance_matrix[edge_add_i]
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
                    gain_diff = TSP.distance_matrix[edge_remove_i] - TSP.distance_matrix[edge_add_i]

                    if node not in neighbors_gain:
                        neighbors_gain[node] = [gain_diff, gain_i]
                        continue
                    if gain_diff > neighbors_gain[node][0]:
                        neighbors_gain[node][0] = gain_diff
                        
        for item in sorted(neighbors_gain.items(), key=lambda node: node[1][0], reverse=True):
            yield item[0], item[1][1]

    def select_edge_remove(self, path, n1, last, gain, edges_remove, edges_add):
        """ Select which edge to remove from the current path """
        if (time.time() - self.start_time) > self.timeLimit:
            return False
            
        neighbor = path.neighbors(last)

        for node_relink in neighbor:
            edge_remove = tuple(sorted([last, node_relink]))
            edge_add = tuple(sorted([node_relink, n1]))
            # Current gain
            gain_i = gain + TSP.distance_matrix[edge_remove]

            # Check for disjointness
            if edge_remove not in edges_add and edge_remove not in edges_remove:
                relink_profit = gain_i - TSP.distance_matrix[edge_add]
                
                # Create shallow copies of both sets and update them
                removed_edges = edges_remove.copy()
                removed_edges.add(edge_remove)
                added_edges = edges_add.copy()
                added_edges.add(edge_add)
                
                # Stop the search when maximum allowed (r-opt-)move was reached
                if len(added_edges) > self.maxR:
                    return False
                
                is_path, new_path = path.create_path(removed_edges, added_edges)

                # Skip if current solution is not a valid path
                if not is_path and len(added_edges) > 2:
                    continue

                # Stop the search when we arrive at an already found solution
                if tuple(new_path) in self.solutions:
                    return False

                # Save the path if it is an improvement
                if is_path and relink_profit > 0:
                    self.Tsp.path = new_path
                    self.Tsp.path_length -= relink_profit
                    if self.doAnalysis:
                        self.analysisData["path_lengths"].append(deepcopy(self.Tsp.path_length))
                        self.analysisData["num_moves"].append(len(added_edges))
                        self.analysisData["elapsed_times"].append(time.time() - self.elapsed_time)
                        self.elapsed_time = time.time()
                    return True
                # Forward removed edges but select another edge to add
                return self.select_edge_add(path, n1, node_relink, gain_i, removed_edges, edges_add)
        return False

    def select_edge_add(self, path, t1, tail, gain_i, edges_remove, edges_add):
        """ Select which edge to add to the current path """
        near = self.nearest_neighbors(tail, path, gain_i, edges_remove, edges_add)
         
        num_nodes = 0
        for node, gain_i in near:
            added_edges = edges_add.copy()
            added_edges.add(tuple(sorted([tail, node])))

            # Forward added edges but select further edge to remove and stop if better path was found
            if self.select_edge_remove(path, t1, node, gain_i, edges_remove, added_edges):
                return True
            else:
                if (time.time() - self.start_time) > self.timeLimit:
                    return False

            num_nodes += 1
            # Stop when enough nodes were tried
            if num_nodes >= self.max_nodes_2p_opt if len(edges_remove) != 2 else self.max_nodes_2opt:
                return False
            
        return False