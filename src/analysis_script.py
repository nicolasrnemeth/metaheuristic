import sys
import json
import numpy as np
import helper
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def path_length(path, node_coors_):
    """ Compute path length """
    node_coors = np.zeros((len(node_coors_[0]), len(node_coors_[1])))
    node_coors[:,0] = np.array(node_coors_[0])
    node_coors[:,1] = np.array(node_coors_[1])
    distance_matrix = squareform( pdist( node_coors, metric="euclidean" ) )
    length = distance_matrix[path[0], path[-1]]
    for idx in range(len(path)-1):
        length += distance_matrix[path[idx], path[idx+1]]
    return length


program, analysisFile, solFile = sys.argv

# Read in optimal solutions
solFile = "input_data/berlin52_optimum"
optimal_path = list()
with open(solFile+".txt", 'r') as sFile:
    for line in sFile.readlines():
        if line.strip() == "EOF":
            break
        if line.strip().isdigit():
            optimal_path.append(int(line.strip())-1)

analysisFile="analysis/berlin52_5_1"
name = analysisFile.split("/")[-1]
    
with open(analysisFile+".json", 'r') as iFile:
    data = json.load(iFile)
    
    
optimal_length = path_length(optimal_path, (data["x_coor"], data["y_coor"]))
times = list(np.cumsum(data["elapsed_times"]))


plt.figure(figsize=(5, 0.18))
plt.box(on=None)
axes = plt.gca()
axes.get_xaxis().set_visible(False)
axes.get_yaxis().set_visible(False)

data_table = [['exec. time', helper.parse_time(data['exec_time'])],
              ['optimal cost', round(optimal_length,2)],
              ['how close is heuristic?', 
               str(round((data['path_lengths'][-1]/optimal_length)*100,2)-100)+" %"],
              ['avg. cost 1000 rand. trials', round(data['avgCost1000Trials'],2)],
              ['# nodes', data['node_count']],
              ['max nodes 2-opt', analysisFile.split("_")[1]],
              ['max nodes r-opt (r > 2)', analysisFile.split("_")[2]]]
color_table = [['beige', 'white']]*7

# Create table
table = plt.table(data_table, color_table, loc='bottom', cellLoc='left')
# Scale, add title and save table
table.scale(1, 1.5)
plt.title("Results overview", y = 0.7, c="k")
plt.savefig("overview_"+name+".pdf", dpi=300, bbox_inches='tight')
plt.close()


# Nodes plot
plt.scatter(data["x_coor"], data["y_coor"], c="green", ec="black", lw=1.5)
plt.box(on=None)
plt.ylim((plt.ylim()[0]-0.05*plt.ylim()[1], plt.ylim()[1]*1.05))
plt.xlim((plt.xlim()[0]-0.05*plt.xlim()[1], plt.xlim()[1]*1.05))
plt.xlabel("x-coordinate", labelpad=10)
plt.ylabel("y-coordinate", labelpad=10)
plt.title(name.split("_")[0])
plt.savefig(name+"_nodes.pdf", dpi=300)
plt.close()


# Path cost over time plot
plt.plot([0]+times, data["path_lengths"], c="r", lw=1.2, zorder=1)
plt.scatter([0]+times, data["path_lengths"], ec="k", s=6, zorder=2, facecolors='none')
plt.plot(plt.xlim(), [optimal_length]*2, lw=1, c="k", linestyle="--")
plt.box(on=None)
plt.ylim((plt.ylim()[0]-0.05*plt.ylim()[1], plt.ylim()[1]*1.05))
plt.xlim((plt.xlim()[0]-0.05*plt.xlim()[1], plt.xlim()[1]*1.05))
plt.xlabel("exec time (in sec.)", labelpad=10)
plt.ylabel("Path Cost", labelpad=10)
plt.title(name.split("_")[0] + " - Cost reduction over time")
plt.savefig(name+"_costOverTime.pdf", dpi=300)
plt.close()


# Number of moves for each improvement
plt.plot(list(range(1,len(data["num_moves"])+1)), data["num_moves"], c="blue")
plt.box(on=None)
plt.ylim((plt.ylim()[0]-0.05*plt.ylim()[1], plt.ylim()[1]*1.05))
plt.xlim((plt.xlim()[0]-0.05*plt.xlim()[1], plt.xlim()[1]*1.05))
plt.xlabel("improvement number", labelpad=10)
plt.ylabel("# moves", labelpad=10)
plt.title(name.split("_")[0] + " - Number of edges removed/added")
plt.savefig(name+"_numMoves.pdf", dpi=300)
plt.close()


# Elapsed time plot
plt.plot(list(range(1,len(data["elapsed_times"])+1)), data["elapsed_times"], c="blue")
plt.box(on=None)
plt.ylim((plt.ylim()[0]-0.05*plt.ylim()[1], plt.ylim()[1]*1.05))
plt.xlim((plt.xlim()[0]-0.05*plt.xlim()[1], plt.xlim()[1]*1.05))
plt.xlabel("improvement number", labelpad=10)
plt.ylabel("Elapsed time (in sec.)", labelpad=10)
plt.title(name.split("_")[0] + " - Elapsed time before next improvement")
plt.savefig(name+"_elapsedTime.pdf", dpi=300)
plt.close()