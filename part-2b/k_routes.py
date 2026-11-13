"""
Finds the K cheapest routes from the origin to a destination using Yen's algorithm (up to 5 routes with their travel times).

Usage:
    from k_routes import k_shortest_paths, print_routes
    
    routes = k_shortest_paths(graph["nodes"], graph["edges"], 970, 4032, method="<assignment 2A method>")
    print_routes(routes)

Usage (command line, tests it on a graph file made by calculate_time.py):
    python k_routes.py <graph file> [method]

method: any of the Assignment 2A methods (DFS, BFS, GBFS, AS, CUS1, CUS2)

NOTE: some methods may not return exactly 5 routes, there may be cases where less than 5 routes are found
"""
import heapq
import sys
import itertools

NUM_ROUTES = 5

# Cost of a route
def route_cost(edges, path):
    total = 0
    for a, b in zip(path, path[1:]):
        total += min(cost for neighbour, cost in edges[a] if neighbour == b)
    return total

# Returns up to 5 different routes, less routes may be returned in the case that there is not that many routes available
def k_shortest_paths(nodes, edges, origin, destination, method="AS"):
    from algorithms import methods
    name = str(method).upper()
    if name not in methods:
        raise ValueError(f"method '{method}' does not exist, methods available: {', '.join(methods)}")
    search = methods[name]
    for site in (origin, destination):
        if site not in nodes:
            raise ValueError(f"site {site} is not in the graph")
    if NUM_ROUTES < 1:
        raise ValueError("Number of routes must be at least 1")
    if origin == destination: # In case the origin IS the destination
        return [(0.0, [origin])]

    # Cheapest path from start to the destination without using the blocked nodes and edges (to get different route), None if there is no path 
    def cheapest(start, blocked_nodes, blocked_edges):
        # Makes a copy of the graph with the blocked nodes and edges taken out
        trimmed = {}
        for a, neighbours in edges.items():
            if a in blocked_nodes: # No edge can start from a blocked node
                continue
            trimmed[a] = []
            for b, cost in neighbours:
                if b in blocked_nodes: # No edge should lead to a blocked node
                    continue
                if (a, b) in blocked_edges: # This edge itself is blocked
                    continue
                trimmed[a].append((b, cost))

        goal, _created, path = search(nodes, trimmed, start, [destination])
        if goal is None: # No path found
            return None
        return path

    # First is the best route (least time needed)
    first = cheapest(origin, set(), set())
    if first is None: # Destination can not be reached at all
        return []
    accepted = [(round(route_cost(edges, first), 6), first)] # Routes confirmed as the 1st, 2nd... cheapest
    candidates = [] # Heap of routes found but not yet confirmed
    seen = {tuple(first)} # Remembers the found routes, prevents repeated routes

    while len(accepted) < NUM_ROUTES:
        last = accepted[-1][1] # Takes in the newest route instead of the cost
        for i in range(len(last) - 1): # Tries every site on the route as the site which the new route splits from
            spur, root = last[i], last[:i + 1] # spur is the splitting site, root is the section kept from the previous route
            blocked_nodes = set(root[:-1]) # Prevents the root's node from being used again (prevent loop)
            
            # Blocks the first road every accepted route with the same root takes, prevents rediscovering an already found route. 'len(p) > i + 1' ensures there is a next step
            blocked_edges = {(path[i], path[i + 1]) for _cost, path in accepted if len(path) > i + 1 and path[:i + 1] == root}
            spur_path = cheapest(spur, blocked_nodes, blocked_edges) # Finds cheapest path from spur to destination under rules set
            if spur_path is not None:
                route = root[:-1] + spur_path # New route is the root with last site removed (spur path starts from it) and spur path appended
                if tuple(route) not in seen: # If path is not 'seen' yet, remember it and add it to candidates with its cost
                    seen.add(tuple(route))
                    heapq.heappush(candidates, (round(route_cost(edges, route), 6), route))
        if not candidates: # No more different routes exist
            break
        accepted.append(heapq.heappop(candidates)) # Cheapest candidate is the next best route
    return sorted(accepted, key=lambda route: route[0]) # Return routes sorted by cost from lowest to highest

# Up to 5 routes starting from the first stop, passing through all of the stops in order
def k_shortest_paths_via(nodes, edges, stops, method="AS"):
    if len(stops) < 2:
        raise ValueError("needs an origin and at least one destination")
    segments = []
    for a, b in zip(stops, stops[1:]): # Pair each stop with the next (e.g 970 -> 4032 -> 4034 turns into (970, 4032) and (4032, 4034))
        segment = k_shortest_paths(nodes, edges, a, b, method=method) # Find up to 5 paths for each segment
        if not segment:
            return []
        segments.append(segment)

    routes = []
    for combination in itertools.product(*segments): # One path from every segment, every combination
        path = list(combination[0][1]) # '0' is the first segment's pair and '1' is its path
        for _cost, segment_path in combination[1:]:
            path += segment_path[1:]
        routes.append((round(sum(cost for cost, _path in combination),6 ), path))
    routes.sort(key=lambda route: route[0])
    return routes[:NUM_ROUTES] # Return up to first 5 joined routes

# Prints the routes, one per line
def print_routes(routes):
    if not routes:
        print("No route found")
    for rank, (cost, path) in enumerate(routes, start=1):
        print(f"Route {rank}: {cost:.2f} minutes, {len(path) - 1} roads: {' -> '.join(map(str, path))}")

# Command line version (for testing)
def main():
    if len(sys.argv) not in (2, 3):
        print("Usage: python k_routes.py <graph file> [method]")
        sys.exit(1)
    from fileparser import parse_file
    method = sys.argv[2] if len(sys.argv) == 3 else "AS" # A* kept as a default method
    try:
        nodes, edges, origin, destinations = parse_file(sys.argv[1])
        stops = [origin, *destinations] # The destinations in the file are visited in the order they are listed
        routes = k_shortest_paths_via(nodes, edges, stops, method=method)
    except FileNotFoundError:
        print(f"Error: file '{sys.argv[1]}' not found!")
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    print(f"Up to {NUM_ROUTES} routes from {' -> '.join(map(str, stops))} using {method.upper()}:")
    print_routes(routes)

if __name__ == "__main__":
    main()