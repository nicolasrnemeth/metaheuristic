# Import packages
import sys
import time
import json
from argparse import ArgumentParser
# Import custom classes
import helper # file with helper functions
from FileParser import FileParser
from LinKernighanHelsgaun import LinKernighanHelsgaun
from TSP import TSP

def _main_(args):
    """ Main program """
    # Parse input file
    fileparser = FileParser(args.filename).parse()
    distance_matrix, coordinates = fileparser.parsed_data()
    
    # Instantiations
    # Specify _rng_ to for reproducibility (initial path is random)
    Tsp = TSP(coordinates, distance_matrix, _rng_=None)
    LinKerHel = LinKernighanHelsgaun(Tsp, args.analysis)
    start_time = time.time()
    # Perform heuristic optimization
    approx_solution, path_length = LinKerHel.optimize()
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

if __name__ == "__main__":
    # Add argument parser
    argParser = ArgumentParser(description="Solve TSP with Lin Kernighan Helsgaun Heuristic.")
    argParser.add_argument("filename", help="Specify name of input file.")
    argParser.add_argument("-o", required=True, type=str)
    argParser.add_argument("--decorate", action="store_true", help="Decorate solution in output file.")
    argParser.add_argument("--stats", action="store_true", help="Print results overview.")
    argParser.add_argument("--analysis", default=None, type=str, help="""Collect analysis data and dump it as JSON file.""")
    
    # Execute main program
    _main_(argParser.parse_args())
    sys.exit(0)