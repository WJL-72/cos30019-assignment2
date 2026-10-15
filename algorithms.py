import math
import sys
import heapq
from collections import deque

# ---------- HELPER FUNCTIONS ----------

# any shared function from your code should be here, helps make it more organised

def sorted_neighbours(edges: dict, node: int) -> list:
    return sorted(edges.get(node, []), key = lambda x: x[0]) # Sorts node neighbours in ascending order (satisfies the sorting requirement)

def path_from(record) -> list:
    path = []
    while record is not None:
        path.append(record[0])
        record = record[1]
    path.reverse()
    return path

# ---------- DFS: DEPTH-FIRST SEARCH ----------
""" Frontier is a stack (LIFO), always expanding the deepest node. Neighbours are pushed in DESCENDING order so that the SMALLEST 
    neighbour is expanded first, thus the ascending-order tie-breaker. Goal test when a node is expanded.
    Not optimal nor guaranteed shortest."""

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

# ---------- BFS: BREADTH-FIRST SEARCH ----------
""" Frontier is a queue (FIFO), always expanding the shallowest node first so the path found has the fewest moves but not lowest cost.
    Neighbours are added in ascending order to satisfy tie-breaking. Goal is tested when a node is expanded.
    Tie-break: shallowest -> smallest node number -> earliest added (queue order).
"""

def bfs(nodes: dict, edges: dict, origin: int, destinations: list):
    goals = set(destinations)

    created = 1  # root node counts as created
    expanded = set() # nodes already expanded, prevent looping on two way edge

    frontier = deque([(origin, None, 0.0)]) # queue of (node, parent record, cost so far), oldest entry is expanded first

    while frontier:
        record = frontier.popleft() # takes shallowest node
        node, _parent, g = record

        if node in expanded: # node added to the queue twice before expanding
            continue

        if node in goals:
            return node, created, path_from(record)

        expanded.add(node)

        for neighbour, cost in sorted_neighbours(edges, node): # ascending order, smallest neighbour joins queue first
            if neighbour in expanded:
                continue

            created += 1
            frontier.append((neighbour, record, g + cost)) # join back of queue

    return None, created, [] # queue exhausted so no destination can be reached.

# ---------- GBFS: GREEDY BEST-FIRST SEARCH ----------
""" Frontier is a priority queue ordered only by h(n) where h(n) is the Euclidean distance to the nearest destination.
    Cost so far is ignored so it is fast but not cost-optimal or fewest moves.
    Tie-break: lowest h -> smallest node number -> earliest added (counter).
"""

def gbfs(nodes: dict, edges: dict, origin: int, destinations: list):
    goals = set(destinations)

    goal_points = [nodes[g] for g in destinations if g in nodes] # coordinates of every destination

    h_cache = {}

    def h(node: int) -> float:
        if node not in h_cache:
            if node in nodes and goal_points:
                h_cache[node] = min(
                    math.dist(nodes[node], point)
                    for point in goal_points
                ) # distance to the nearest destination
            else:
                h_cache[node] = 0.0 # no coordinates so no guide

        return h_cache[node]

    counter = 0 # insertion order (final tie-breaker)
    created = 1 # root node counts as created
    expanded = set() # nodes already expanded, prevent looping on two-way edge

    root = (origin, None, 0.0)

    frontier = [(h(origin), origin, counter, root)] # heap items: (h, node number, counter, record)

    while frontier:
        _h, node, _count, record = heapq.heappop(frontier) # lowest h first

        if node in expanded: # node added to heap twice before expanded (stale)
            continue

        if node in goals: # goal test during expansion
            return node, created, path_from(record)

        expanded.add(node)

        g = record[2] # cost so far (tracked only to show on output)

        for neighbour, cost in sorted_neighbours(edges, node):
            if neighbour in expanded:
                continue

            created += 1
            counter += 1

            heapq.heappush(
                frontier,
                (
                    h(neighbour),
                    neighbour,
                    counter,
                    (neighbour, record, g + cost)
                )
            )

    return None, created, [] # heap exhausted, no destination can be reached.

# ---------- AS: A* SEARCH ----------
""" Frontier is a priority queue ordered by f(n) = g(n) + h(n), where g = cost so far, and h = straight-line (euclidean) distance to the
    NEAREST destination to support multiple destinations. Goal test when a node is expanded/popped. Guarantees cheapest path so long h is not
    overestimated.
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
""" Works by setting a limit on the score threshold, f(n) = g(n) + h(n). If a goal is not found within the limit set, raise limit and search again.
    g = number of moves so far, h = estimated number of moves remaining (straight-line distance / longest edge)
    Tie-breaking: fewest moves -> lowest cost -> smallest node number"""

def cus2(nodes: dict, edges: dict, origin: int, destinations: list):
    goals = set(destinations)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), len(nodes) + 100)) # Setting limit for recursion (path cannot be longer than no of nodes)

    goal_points = [nodes[g] for g in destinations if g in nodes]  # Sets coordinates (x, y) of every goal

    # Loop through all edges to find the longest edge (cost not used since this prioritises SHORTEST path)
    longest_edge = 0.0 
    for start, neighbours in edges.items():
        for end, _cost in neighbours:
            if start in nodes and end in nodes:
                longest_edge = max(longest_edge, math.dist(nodes[start], nodes[end]))

    # Calculating heuristic
    def h(node: int) -> int:
        if node in nodes and goal_points and longest_edge > 0:
            nearest = min(math.dist(nodes[node], point) for point in goal_points) # Straight-line distance to the nearest goal/destination
            return math.ceil(nearest / longest_edge - 1e-9) # Gets edges that are available to move (- 1e-9 prevents float rounding errors)
        return 0 # If there is no information, do not move from current node

    # Shared search state across iterations
    path = [origin] # Returns currently explored branch
    on_path = {origin} # Check if the node is in current path
    created = 0 # Total search-tree nodes created across all iterations (cumulative)
    best_cost = math.inf # Cost of cheapest path so far (inf so any number will be smaller)
    best_path = None # No path until a goal is reached

    def limited_dfs(g: int, cost_so_far: float, limit: int):
        """ Depth-first search from the last node of 'path' while not going past f = limit, returns the smallest f that was cut off
            g - moves made so far, cost_so_far is the total accumulated cost
            Goals reached are stored in best_cost/best_path """
        nonlocal created, best_cost, best_path # Ensures the variable are updated rather than creating new ones
        node = path[-1] # Current node is last one on path
        f = g + h(node) # Estimate = moves-so-far + estimated moves-to-go

        # Checks if the threshold is hit (no goal found) or if goal is found
        if f > limit:
            return f
        if node in goals:
            if cost_so_far < best_cost: # Strictly cheaper only, so an equally cheap path found later does not replace the first one
                best_cost = cost_so_far
                best_path = list(path)
            return math.inf # No need to go past a goal

        # Tracks smallest f value rejected under this node
        smallest_cut = math.inf
        for neighbour, cost in sorted_neighbours(edges, node):
            if neighbour in on_path: # Skips nodes which are already on the search branch (prevent loop on two-way edges)
                continue

            new_cost = round(cost_so_far + cost, 9)
            if new_cost >= best_cost: # Skip if the cost is higher than lowest
                continue

            # Creates new search tree node
            created += 1
            path.append(neighbour)
            on_path.add(neighbour)

            smallest_cut = min(smallest_cut, limited_dfs(g + 1, new_cost, limit)) # Updates the smallest f value after increasing recursion by 1

            # Remove path without goal
            path.pop()
            on_path.remove(neighbour)

        return smallest_cut # End-case if no goal was found, return smallest f value refused

    # Iterations
    limit = h(origin) # First limit will be h(origin)
    while limit != math.inf: # Check for infinite limit (no solution)
        created += 1  # Rebuilds search tree on every iteration, restarts search with 0 moves
        next_limit = limited_dfs(0, 0, limit)
        if best_path is not None: # This iteration reached a goal and has fewest moves so far
            return best_path[-1], created, best_path
        limit = next_limit # Raise budget to the smallest f value rejected

    return None, created, [] # Usually skipped if there is a solution to the test case, will only reach here if NO SOLUTIONS are available

# Toggle to make the agent visit all destinations
def visit_all(algorithm):
    def run(nodes, edges, origin, destinations): # Overwrites the run code
        remaining, path, created, goal = set(destinations), [origin], 0, None
        while remaining: # Keeps algorithm running until all destinations are visited
            found, count, segment = algorithm(nodes, edges, origin, sorted(remaining)) # Runs the search from current origin
            created += count
            if found is None: # Rest of the nodes are unreachable, stop searching
                break
            path += segment[1:] # Segment starts at the old origin so skip it
            remaining.discard(found) # Removes found goal from the list
            origin = goal = found # Update the goal reached to be new origin
        return goal, created, (path if goal is not None else [])
    return run

# ---------- METHODS DICTIONARY ----------
methods: dict = { # Used for listing all methods in search.py
    'DFS': dfs,
    'BFS': bfs,
    'GBFS': gbfs,
    'AS': a_star,
    'CUS2': cus2
}