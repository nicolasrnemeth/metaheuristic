# Import required packages
import sys
import numpy as np
from os import path
from scipy.spatial.distance import squareform, pdist


# Custom class
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