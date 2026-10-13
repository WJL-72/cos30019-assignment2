import math
import sys
import heapq
from collections import deque

# ---------- HELPER FUNCTIONS ----------

# any shared function from your code should be here, helps make it more organised

def sorted_neighbours(edges: dict, node: int) -> list:
    return sorted(edges.get(node, []), key = lambda x: x[0]) # Sorts node neighbours in ascending order (satifies the sorting requirement)

def path_from(record) -> list:
    path = []
    while record is not None:
        path.append(record[0])
        record = record[1]
    path.reverse()
    return path

# ---------- DFS: DEPTH-FIRST SEARCH ----------
""" Frontier is a stack (LIFO), always expanding the deepest node. Neighbours are pushed in DESCENDING order so that the SMALLEST neighbour
    is expanded first, thus the ascending-order tie-breaker. Goal test when a node is expanded. Not optimal nor guaranteed shortest."""

def dfs(nodes: dict, edges: dict, origin: int, destinations: list):
    goals = set(destinations)
    created = 1 # root node counts as created
    expanded = set() # nodes already expanded
    frontier = [(origin, None, 0.0)] #stack, last item expanded next

    while frontier:
        record = frontier.pop()
        node, _parent, g = record

        if node in expanded: # node was added to stack twice before expanding (stale)
            continue
        if node in goals: # goal test in expansion
            return node, created, path_from(record)
        expanded.add(node) #mark as expanded so it doesnt expand again, avoids loop

        for neighbour, cost in reversed(sorted_neighbours(edges, node)): # reversed so that smallest ends on top of stack
            if neighbour in expanded:
                continue
            created += 1
            frontier.append((neighbour, record, g+cost))

    return None, created, [] # frontier exhausted, no dest

# ---------- AS: A* SEARCH ----------
""" Frontier is a priority queue ordered by f(n) = g(n) + h(n), where g = cost so far, and h = straight-line (euclidean) distance to the
    NEAREST destination to support multiple destinations. Goal test when a node is expanded/popped. Guarantees cheapest path so long h is not
    estimated.
    Tie-break: lowest f -> smallest node number -> earliest added (counter). """

def a_star(nodes: dict, edges: dict, origin: int, destinations: list):
    goals = set(destinations)
    goal_points = [nodes[g] for g in destinations if g in nodes] # coordinates every dest

    h_cache = {} # h depends on the node, so it computes once
    def h(node: int) -> float:
        if node not in h_cache:
            if node in nodes and goal_points:
                h_cache[node] = min(math.dist(nodes[node], point) for point in goal_points)
            else:
                h_cache[node] = 0.0 # no coords, falls back to uniform-cost
        return h_cache[node]

    def f_key(g: float, node: int) -> float:
        return round(g + h(node), 9)

    counter = 0 # insertion order, final tie breaker
    created = 1 # root node counts as created
    expanded = {} # node -> cheapest g it was expanded with
    root = (origin, None, 0.0)
    frontier = [(f_key(0.0, origin), origin, counter, root)] # heap items

    while frontier:
        _f, node, _count, record = heapq.heappop(frontier)
        g = record[2]

        if node in expanded and expanded[node] <= g: # already expanded via equal or cheaper route
            continue
        if node in goals: # goal test on expansion (optimal)
            return node, created, path_from(record)
        expanded[node] = g

        for neighbour, cost in sorted_neighbours(edges, node):
            new_g = g+cost
            if neighbour in expanded and expanded[neighbour] <= new_g:
                continue
            created += 1
            counter += 1
            heapq.heappush(frontier, (f_key(g+cost, neighbour), neighbour, counter, (neighbour, record, new_g)))

    return None, created, []

# ---------- CUS2: IDA* (ITERATIVE DEEPENING A*) ----------
""" Works by setting a limit on the score threshold, f(n) = g(n) + h(n). If a goal is not found within the limit set, raise limit and search again
    IMPORTANT NOTE: g COUNTS MOVES, NOT COST, so the main aim is the route with the FEWEST MOVES (unlike normal A*, which finds the lowest cost).
    Tie-break: if several routes have the same (fewest) number of moves, the one with the LOWEST TOTAL COST is chosen, as A* would prefer. To do this
    the last iteration does not stop at the first goal, it checks every route within the limit, and skips any route that costs more than the best
    one found so far. If costs are also equal, the route ending at the SMALLER goal node number wins (same as A*), then the first route found. """

def cus2(nodes: dict, edges: dict, origin: int, destinations: list):
    goals = set(destinations)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), len(nodes) + 100)) # Setting limit for recursion (path cannot be longer than no of nodes)

    # Heuristics for the search
    goal_points = [nodes[g] for g in destinations if g in nodes]  # Sets coordinates (x, y) of every goal

    # Loop through all edges to find the longest edge (cost not used since this proritises SHORTEST path)
    longest_edge = 0.0 
    for start, neighbours in edges.items():
        for end, _cost in neighbours:
            if start in nodes and end in nodes:
                longest_edge = max(longest_edge, math.dist(nodes[start], nodes[end]))

    h_cache = {}  # Stores h value so does not need recalculation over multiple iterations

    def h(node: int) -> int:
        if node not in h_cache:
            if node in nodes and goal_points and longest_edge > 0:
                nearest = min(math.dist(nodes[node], point) for point in goal_points)
                h_cache[node] = math.ceil(nearest / longest_edge - 1e-9) # Gets edges that are available to move (- 1e-9 prevents float rounding errors)
            else:
                h_cache[node] = 0  # If there is no information (no coordinates, no goal found or no length edge), do not move from the current node
        return h_cache[node]

    # Shared search state across iterations
    path = [origin] # Returns currently explored branch
    on_path = {origin} # Check if the node is in current path
    created = 0 # Total search-tree nodes created across all iterations (cumulative)
    best_cost = math.inf # Total cost of the cheapest goal route found so far in this iteration
    best_path = None # That cheapest route

    def limited_dfs(g: int, cost: float, limit: int):
        """ Depth-first search from the last node of 'path' while not going past f = limit. g = moves so far, cost = total edge cost so far.
            Returns the smallest f that was cut off (math.inf if nothing was cut off). Goal routes are saved in best_path / best_cost """
        nonlocal created, best_cost, best_path # Ensures the variables are updated rather than creating new ones
        node = path[-1] # Current node is last one on path
        f = g + h(node) # Estimate = moves-so-far + estimated moves-to-go

        # Checks if the threshold is hit (no goal found), if this route cannot beat the best one, or if goal is found
        if f > limit:
            return f
        if cost > best_cost + 1e-9: # Already costs more than the best route found (1e-9 prevents float rounding errors), so it cannot be better
            return math.inf
        if node in goals: # Keep this route if it is cheaper, or the same cost but ends at a smaller goal number (like A*). Otherwise the first route found stays
            if cost < best_cost - 1e-9 or (best_path is not None and node < best_path[-1]):
                best_cost = cost
                best_path = list(path)
            return math.inf

        # Tracks smallest f value
        smallest_cut = math.inf
        for neighbour, edge_cost in sorted_neighbours(edges, node):
            if neighbour in on_path: # Skips nodes which are already on the search branch (prevent loop on two-way edges)
                continue

            # Creates new search tree node
            created += 1
            path.append(neighbour)
            on_path.add(neighbour)

            # Increases the recursion threshold (1 more step), keeps the smallest f that was cut off
            smallest_cut = min(smallest_cut, limited_dfs(g + 1, cost + edge_cost, limit))

            # Remove the node from the branch (whether or not it led to a goal, since every route in the limit is checked)
            path.pop()
            on_path.remove(neighbour)

        return smallest_cut # End-case, return smallest F that was cut off

    # Iterations
    limit = h(origin) # First limit will be h(origin)
    while limit != math.inf: # Check for infinite limit (no solution)
        created += 1  # Rebuilds search tree on every iteration, restarts search with 0 moves
        cut = limited_dfs(0, 0.0, limit)
        if best_path is not None: # A goal was reached and every route within the limit has been checked, so best_path is the cheapest fewest-move route
            return best_path[-1], created, best_path
        limit = cut # Raise budget to the smallest f value rejected

    return None, created, [] # Usually skipped if there is a solution to the test case, will only reach here if NO SOLUTIONS are available

# ---------- METHODS DICTIONARY ----------
methods: dict = { # Used for listing all methods in search.py
    'DFS': dfs,
    'AS': a_star,
    'CUS2': cus2
}