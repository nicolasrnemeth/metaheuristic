# Import required packages
import sys
import numpy as np
from os import path
from scipy.spatial.distance import squareform, pdist


# Custom class
class FileParser(object):
    """ Parse TSP input file """
    def __init__(self, filepath):
        self.filepath = filepath
        self.distMatrix = None
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
            self.distMatrix = squareform( pdist( nodes, metric="euclidean" ) )

    def parse(self):
        if not path.exists(self.filepath):
            print("Inputfile could not be found, please check filename and path.")
            sys.exit(0)
            
        with open(self.filepath, 'r') as ifile:
            for idx, line in enumerate(ifile.readlines()):
                if line.strip() == "NODE_COORD_SECTION" or line.strip() == "DISPLAY_DATA_SECTION":
                    self.node_coord(idx)
                    return self
            if line.strip().split(" ")[0].isdigit():
                print("The input file is in the wrong format (see README.md for format specs).")
                sys.exit(0)
        return self
    
    # Return parsed data
    def parsed_data(self):
        return self.distMatrix, self.coordinates