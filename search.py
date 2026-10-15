import sys
import time
import tracemalloc
from fileparser import parse_file
from algorithms import methods, visit_all

memory_test_limit = 4000 # Stops memory tracking if a search takes longer than 4 seconds, tracing the memory would make it even slower

# Function to calculate route cost
def route_cost(edges: dict, path: list) -> float:
    total = 0
    for a, b in zip(path, path[1:]): # Creates node pairs
        total += min(cost for neighbour, cost in edges[a] if neighbour == b) # Keeps cost of selected neighbour and adds it to total
    return total

# Measures the statistics of the search
def run_measured(algorithm, nodes, edges, origin, destinations):
    start = time.perf_counter()
    goal, node_count, path = algorithm(nodes, edges, origin, destinations) # Runs selected algorithm
    time_taken = (time.perf_counter() - start) * 1000

    if time_taken < memory_test_limit: # Calculates memory usage if search is less than 4 seconds
        tracemalloc.start()
        algorithm(nodes, edges, origin, destinations)
        peak_memory = tracemalloc.get_traced_memory()[1]/1000 # Gets highest recorded memory usage of the search
        tracemalloc.stop()
        memory_text = f"{peak_memory:.1f} kB"
    else:
        memory_text = f"Not measured (search took over {memory_test_limit / 1000:g} s)"

    return goal, node_count, path, time_taken, memory_text    

def main():
    if len(sys.argv) not in (3, 4) or (len(sys.argv) == 4 and sys.argv[3].lower() != '--all'): # Validates 3 arguments or 4 with the optional --all toggle
        print("Usage: python search.py <filename> <method> [--all]")
        print(" --all: visits ALL destinations instead of stopping after one")
        print(f"Methods available: {', '.join(methods)}")
        sys.exit(1)

    filename = sys.argv[1]
    method = sys.argv[2].upper()
    visit_every = len(sys.argv) == 4 # True when --all is typed

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
        print(f"Error while parsing '{filename}': {exc}")
        sys.exit(1)

    algorithm = visit_all(methods[method]) if visit_every else methods[method] # Wraps the method if all destinations must be visited
    goal, node_count, path, time_taken, memory_text = run_measured(algorithm, nodes, edges, origin, destinations)

    print(f"Test case file: {filename}")
    print(f"Method selected: {method}" + (" (visit all)" if visit_every else ""))
    print(f"Time taken for test: {time_taken:.3f} ms")
    print(f"Memory usage (peak): {memory_text}")

    if goal is None:
        print("Goal reached: None (no destination is reachable)")
        print(f"Nodes created: {node_count}")
        return

    if visit_every:
        order = []
        for n in path: # Goals in the order they were first reached
            if n in destinations and n not in order:
                order.append(n)
        missed = [d for d in dict.fromkeys(destinations) if d not in order] # Any unreachable destination will be added here
        print(f"\nGoals visited (in order): {' -> '.join(map(str, order))}")
        if missed:
            print(f"Goals not reached (no route from last goal visited): {', '.join(map(str, missed))}")
    print(f"\nGoal reached: Node {goal}")
    print(f"Nodes created: {node_count}")
    print(f"Number of moves: {len(path) - 1}" )
    print(f"Total cost: {route_cost(edges, path):g}")
    print("Final route:")
    print(' -> '.join(str(n) for n in path))

if __name__ == '__main__':
    main()