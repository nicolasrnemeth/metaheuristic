class LinKernighanHelsgaun(object):
    def __init__(self, Tsp, doAnalysis=False):
        self.solutions = set()
        self.neighbors  = dict()
        self.Tsp = Tsp
        self.doAnalysis = True if doAnalysis else False
    
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
        if len(edges_remove) == 4:
            prev, next_ = route.neighbors(last)

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
                    choice = self.chooseY(route, t1, node_relink, gain_i, removed, edges_add)

                    if len(edges_remove) == 2 and choice:
                        return True
                    else:
                        # Single iteration for i > 2
                        return choice

        return False

    def chooseY(self, route, t1, tail, gain_i, edges_remove, edges_add):
        ordered = self.closest(tail, route, gain_i, edges_remove, edges_add)

        if len(edges_remove) == 2:
            # Check the five nearest neighbours when i = 2
            top = 5
        else:
            # Otherwise the closest only
            top = 1

        for node, (_, gain_i) in ordered:
            edge_add = tuple(sorted([tail, node]))
            added = deepcopy(edges_add)
            added.add(edge_add)

            # Stop at the first improving tour
            if self.chooseX(route, t1, node, gain_i, edges_remove, added):
                return True

            top -= 1
            # Tried enough options
            if top == 0:
                return False

        return False