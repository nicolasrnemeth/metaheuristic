## Heuristic Algorithm for solving the TSP problem

- author: Nicolas Nemeth

- Problem: Travelling Salesman Problem

- Approach: Neighborhood Search with R-opt first improvement heuristic

## Dependencies

--> see the file "dependencies.txt"

## Hardware:
- Intel Core i5-1135G7 CPU @ 2.40GHz, 16 GB RAM

## Instructions to run the code: all commands were written as if executed from the root of the folder

### For branchNbound --> my own implementation

1. Enter following into the terminal:

python src/__main__.py input_data/berlin52.txt -o solution.txt

more generally (input_filename and -o output filename are required):

python src/__main__.py input_filename.txt -o output_filename.txt

Flags/Options:
--params                        choose how many of the most promising neighbors should be tried for r = 2 and r > 2
                                (input should be provided as two consecutive integers, e.g. "--params 5 1" )
                                --> --params sets max_nodes_2opt and max_nodes_2p_opt variables in the VarNeighSearch class
--decorate                      decorates the solution output for better readability
--stats                         print execution time, the path length and the found path
--timeLimit                     set a time limit in minutes after which the optimization should stop 
                                and return best solution until this moment (e.g. "--timeLimit 2" --> after 2 minutes stop)
--analysis name_of_output_file     collect analysis data during the optimization run 
                                   and output the data to a json file called name_of_output_file
--maxOptMove                    set the maximum number of opt-moves which are allowed to be performed to improve current path
                                in other words set maximum r in r-opt heuristic (e.g. "--maxOptMove 6" --> no more than 6-opt-moves are allowed) 


### Plots and analysis
The plots were created using the python script "analysis_script.py".
It is just provided as a reference, without detailed explanation how to use it.
Basically, it requires an analysis file created by the --analysis option above and a file containing the optimal solution (in a certain format --> see the folder "input_data" --> files which have "optimum" in their name).


## Remarks

- "src" folder contains all files which are part of the implementation
  
- "solutions" folder contains the solutions as obtained by the optimisation runs

- "plots" folder contains the plots corresponding to the analysis as obtained by the optimization runs

- "input_data" contains the input data files and the optimal solutions

- "analysis" contains the json files containing the collected analysis data from the optimization runs


## Format of input files

- the input files provided to the __main__.py script must have the following format

NODE_COORD_SECTION
1 565.0 575.0
2 25.0 185.0
3 345.0 750.0
EOF

The FileParser class in FileParser.py will start reading in the node coordinates from the input file,
when it has encountered "NODE_COORD_SECTION" (case sensitive).

The first number is the index of the node the second and third must be the coordinates.
The node coordinate section must end with "EOF" to tell the FileParser when to stop reading in.
Everything before "NODE_COORD_SECTION" and after "EOF" that is written in the input file will be ignored.

When the input file also contains a computed edge weight section and the node coordinates are indicated with
"DISPLAY_DATA_SECTION" and "EOF", instead of "NODE_COORD_SECTION" and "EOF" as in input file gr120.txt then the FileParser
will start reading in the node coordinates after "DISPLAY_DATA_SECTION" and stop at "EOF".
The input file must contain node coordinates. It does not work if only edge weights are provided. The FileParser computes
the distance matrix already from the node coordinates.

## Sources 
(following sources were used to gain insights and ideas on how to implement the heuristic)

- https://en.wikipedia.org/wiki/2-opt
- http://www.ra.cs.uni-tuebingen.de/lehre/ss03/vs_mh/mh-vorlesung.pdf
- https://doi.org/10.1007/s12532-009-0004-6
