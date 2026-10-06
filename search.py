import sys
import time
from fileparser import parse_file
from algorithms import methods

# Function to calculate route cost
def route_cost(edges: dict, path: list) -> float:
    total = 0
    for a, b in zip(path, path[1:]): # Creates node pairs
        total += min(cost for neighbour, cost in edges[a] if neighbour == b) # Keeps cost of selected neighbour and adds it to total
    return total

def main():
    if len(sys.argv) != 3: # Validates 3 arguments are given
        print("Usage: python search.py <filename> <method>")
        print(f"Methods available: {', '.join(methods)}")
        sys.exit(1)

    filename = sys.argv[1]
    method = sys.argv[2].upper()

    if method not in methods: # Error if method selected does not exist
        print(f"Error: '{method}' does not exist!")
        print(f"Methods available: {', '.join(methods)}")
        sys.exit(1)

    # Error handling for file parsing
    try:
        nodes, edges, origin, destinations = parse_file(filename)
    except FileNotFoundError:
        print(f"Error: file '{filename}' not found!")
        sys.exit(1)
    except Exception as exc:
        print(f"Error while parsing '{filename}': {exc}") # wow im using try and exceptions just like what Dr Fu taught
        sys.exit(1)

    start = time.perf_counter()
    goal, node_count, path = methods[method](nodes, edges, origin, destinations) # Runs selected algorithm
    time_taken = (time.perf_counter() - start) * 1000

    print(f"Test case file: {filename}")
    print(f"Method selected: {method}")
    print(f"Time taken for test: {time_taken:.3f} ms")

    if goal is None:
        print("Goal reached: None (no destination is reachable)")
        print(f"Nodes expanded: {node_count}")
        return

    print(f"\nGoal reached: Node {goal}")
    print(f"Nodes expanded: {node_count}")
    print(f"Length of route: {len(path) - 1}" )
    print(f"Total cost: {route_cost(edges, path):g}")
    print("Final route:")
    print(' -> '.join(str(n) for n in path))

if __name__ == '__main__':
    main()