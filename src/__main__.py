# Import packages
import sys
import time
import json
from numpy import inf as Infinity
from argparse import ArgumentParser
# Import custom classes
import helper
from TSP import TSP
from FileParser import FileParser
from VarNeighSearch import VarNeighSearch

def _main_(args):
    """ Main program """
    # Parse input file
    fileparser = FileParser(args.filename).parse()
    distance_matrix, coordinates = fileparser.parsed_data()
    
    # Instantiations
    # Specify _rng_ to for reproducibility (initial path is random)
    Tsp = TSP(coordinates, distance_matrix, _rng_=None)
    LocalSearch = VarNeighSearch(Tsp, args.params[0], args.params[1], args.analysis, args.timeLimit, args.maxOptMove)
    start_time = time.time()
    # Perform heuristic optimization
    approx_solution, path_length, analysisData = LocalSearch.optimize()
    # Parse execution time
    exec_time = helper.parse_time( time.time() - start_time )
    
    # Write approximate solution to file
    helper.output_solution(approx_solution, args.o, args.decorate)
    
    # Print results overview
    if args.stats:
        print("\nRunning_time:", exec_time)
        print("Length of heuristic path (rounded to 1 digit after decimal point) =", round(path_length,1))
        print("\nPath found by heuristic:\n")
        for node in approx_solution:
            print(str(node)+' ->', end=' ')
        print(approx_solution[0])
        print("\n")
        
    # Write analysis data to json file
    if args.analysis is not None:
        json_string = json.dumps(analysisData, indent=4)
        with open(args.analysis+'.json', 'w') as dataFile:
            dataFile.write(json_string)

if __name__ == "__main__":
    # Add argument parser
    argParser = ArgumentParser(description="Solve TSP with R-opt Heuristic.")
    argParser.add_argument("filename", help="Specify name of input file.")
    argParser.add_argument("-o", required=True, type=str)
    argParser.add_argument("--params", default=[5,1], type=int, nargs=2, 
                           help="Change default values of parameters. (see README.txt for explanations)")
    argParser.add_argument("--decorate", action="store_true", help="Decorate solution in output file.")
    argParser.add_argument("--stats", action="store_true", help="Print results overview.")
    argParser.add_argument("--analysis", default=None, type=str, help="Collect analysis data and dump it as JSON file.")
    argParser.add_argument("--timeLimit", default=Infinity, type=float, 
                           help="Set a time limit (in minutes) after which to stop optimization run.")
    argParser.add_argument("--maxOptMove", default=Infinity, type=float,
                           help="Set the maximum number of moves (k) which are allowed, to find a better solution in the r-neighborhood of the current.")
    
    # Execute main program
    _main_(argParser.parse_args())
    sys.exit(0)