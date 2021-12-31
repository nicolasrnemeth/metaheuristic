# Import required packages
import time
from numpy import inf as Infinity
from TSP import TSP, Path

# Custom class
class LinKernighan(object):
    """ Heuristic to solve TSP """
    def __init__(self, Tsp, doAnalysis=False, timeLimit=Infinity):
        self.Tsp = Tsp
        self.timeLimit = timeLimit*60
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
                    edges_add = set([tuple(sorted([n2, n3]))])
                    
                    # Restart the algorithm after a better path was found
                    if self.select_edge_remove(path, n1, n3, gain_i, edges_remove, edges_add):
                        return True
                    
                    num_search += 1
                    # Set number of searches to perform before to try a new node n2
                    if num_search == 5:
                        break
        return False
    
    def optimize(self):
        """ Start heuristic optimization """
        start_time = time.time()
        while (time.time() - start_time) < self.timeLimit:
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
                        neighbors_gain[node][0] = gain_diff
                    else:
                        neighbors_gain[node] = [gain_diff, gain_i]
                        
        neighbors_gain = sorted(neighbors_gain.items(), key=lambda x: x[1][0], reverse=True)
        return neighbors_gain

    def select_edge_remove(self, path, n1, last, gain, edges_remove, edges_add):
        """ Select which edge to remove from the current path """
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
                if tuple(new_path) in self.solutions:
                    return False

                # Save the path if it is an improvement
                if is_path and relink_profit > 0:
                    self.Tsp.path = new_path
                    self.Tsp.path_length -= relink_profit
                    return True
                else:
                    # Forward removed edge but select another edge to add
                    selection_add = self.select_edge_add(path, n1, node_relink, gain_i, removed, edges_add)

                    if len(edges_remove) == 2 and selection_add:
                        return True
                    else:
                        return selection_add
        return False

    def select_edge_add(self, path, t1, tail, gain_i, edges_remove, edges_add):
        """ Select which edge to add to the current path """
        near = self.nearest_neighbors(tail, path, gain_i, edges_remove, edges_add)
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
            # Stop when maximum number of searches reached
            if num_search == 0:
                return False
            
        return False