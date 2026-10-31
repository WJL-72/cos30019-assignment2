COS30019 Assignment 2A - Tree Based Search

A Python program that solves the Route Finding problem - finding a path from the origin to one or all of the destination nodes on a 2D directed and weighted graph.
Six algorithms were created along with a command-line interface, a GUI visualiser and a batch testing file.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

1. Requirements

-Python 3.12 or newer (program was tested using Python 3.12.10 and 3.14.8)
-Windows 10 or 11 operating system
-No external libraries were used (tkinter is shipped with Python installation and is used for the GUI)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

2. File structure

Keep all of the files below in the same folder:

search.py - Command-line version of the program
algorithms.py - All of the search algorithms along with the '--all' toggle (search all destination nodes)
fileparser.py - Reads and extracts information from a problem file (has error handling)
gui.py - GUI version of search.py which animates the search
testgui.py - Draws out the graph only (does not perform search), useful for visualising test cases for checking
batchtest.py - Runs all test cases with all methods and prints the results to results.csv
test_cases/ - Folder to store problem files

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

3. How to run

Open a terminal/command prompt in the project folder and run the following command:

**Command Line Version**
python search.py <filename> <method> [--all]

- <filename>: A problem file to be solved, can be a full or relative path or just the name of the file in test_cases/.
- <method>: One of the methods below:
    - DFS: depth-first search
    - BFS: breadth-first search
    - GBFS: greedy best-first search
    - AS: A* search ("A Star")
    - CUS1: bidirectional flood fill search
    - CUS2: iterative deepening A* search (IDA*)

- --all: Optional argument. Visits all reachable destinations if used instead of stopping at first destination.

Examples:
python search.py testcase-1.txt DFS
python search.py test_cases/testcase-7.txt CUS2 --all

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**GUI version**
Use the exact same arguments as CLI but switch out 'search.py' for 'gui.py':

python gui.py <filename> <method> [--all]

Examples:
python search.py testcase-3.txt GBFS
python search.py test_cases/testcase-4.txt CUS1 --all

When the program is run using the GUI version, a Tkinter window will pop up and animate the graph, along with a legend for node states. A summary is shown when the search concludes.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

To view a graph without searching, use the following command:
python testgui.py <filename> ; where filename is the problem file to visualise

To run the batch tester to get results for all methods on all test cases, run the following command:
python batchtest.py

All results will be saved to a results.csv file with information of the search such as time taken, peak memory, goal reached, nodes created, moves, cost and route found.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

4. Example of input file format:

Nodes
1: (1,2)
2: (3,2)
3: (5,3)

Edges:
(1,2): 3
(2,3): 3

Origin:
1

Destinations:
3

Node: one line per node as id: (x, y)
Edges: one line per directed edge as (from, to): cost
Origin: the node where the search starts
Destinations: one or more nodes separated using a semicolon (;)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

5. Expected output for CLI version:

Test case file: testcase-1
Method selected: CUS2
Time taken for test: 0.068 ms
Memory usage (peak): 3.0 kB

Goal reached: Node 2
Nodes created: 9
Number of moves: 2
Total cost: 80
Final route:
1 -> 3 -> 2

Goal reached: the destination found by the algorithm
Nodes created: nodes added to the search tree (for CUS1, both directions are counted. for CUS2, it is cumulative across all iterations)
Memory usage (peak): measured by running the search for a second time using tracemalloc. It is skipped if the search takes more than 4 seconds to complete


Important notes:
- If no destination can be reached, it prints out "Goal reached: None (no destination is reachable)" with the number of nodes created.
- With --all, the output will list all of the destinations visited in order and any that could not be reached.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

6. Tie-breaking

Each algorithm have their own way of tie-breaking but when all else are equal, the lower numbered node is expanded first and then the node which was added earlier. Neighbour nodes will always be considered in ascending order of node ID.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

7. The'--all' option

When a search is executed with this option, the search will run repeatedly. When it reaches a destination from the first origin, it will update the destination as the new origin and search for the next destination, joining all search segments into one final route. This is a simpler and faster way of implementing this but not guaranteed to give the shortest possible path and destinations may be missed if the agent enters a dead end through a one-way edge (example: testcase-7 with AS, CUS1 or CUS2)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

8. Known limitations

- There is no timeout so searches run until they find a destination or decides no destination is reachable.
- CUS2 (IDA*) slows down tremendously on large graphs even if the destination is unreachable.
- Peak memory is approximate as it only counts the amount allocated by Python on a second run of the search
- The GUI only records up to 15 seconds (~750 frames) so very long searches will skip some steps