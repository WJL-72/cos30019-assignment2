import math
import sys
import heapq
from collections import deque

# ---------- HELPER FUNCTIONS ----------
def sorted_neighbours(edges: dict, node: int) -> list:
    return sorted(edges.get(node, []), key = lambda x: x[0]) # Sorts node neighbours in ascending order (satisfies the sorting requirement)

def path_from(record) -> list:
    path = []
    while record is not None:
        path.append(record[0])
        record = record[1]
    path.reverse()
    return path

def euclidean_heuristic(nodes: dict, destinations: list): # Builds heuristics for GBFS and A*, straight line distance to nearest destination
    goal_points = [nodes[g] for g in destinations if g in nodes] # Coordinates of every destination
    h_cache = {}

    def h(node: int) -> float: # h depends on the node, so it computes once a node
        if node not in h_cache:
            if node in nodes and goal_points:
                h_cache[node] = min(math.dist(nodes[node], point) for point in goal_points) # Distance to the nearest destination
            else:
                h_cache[node] = 0.0 # No coordinates so no guide (h = 0, A* becomes uniform cost while GBFS falls back to node number order)
        return h_cache[node]

    return h

# ---------- DFS: DEPTH-FIRST SEARCH ----------
""" Frontier is a stack (LIFO), always expanding the deepest node. Neighbours are pushed in DESCENDING order so that the SMALLEST 
    neighbour is expanded first, thus the ascending-order tie-breaker. Goal test when a node is expanded.
    Not optimal nor guaranteed shortest."""

def dfs(nodes: dict, edges: dict, origin: int, destinations: list):
    goals = set(destinations)
    created = 1 # root node counts as created
    expanded = set() # nodes already expanded
    frontier = [(origin, None, 0.0)] # stack, last item expanded next

    while frontier:
        record = frontier.pop()
        node, _parent, g = record

        if node in expanded: # node was added to stack twice before expanding (stale)
            continue
        if node in goals: # goal test in expansion
            return node, created, path_from(record)
        expanded.add(node) # mark as expanded so it doesnt expand again, avoids loop

        for neighbour, cost in reversed(sorted_neighbours(edges, node)): # reversed so that smallest ends on top of stack
            if neighbour in expanded:
                continue
            created += 1
            frontier.append((neighbour, record, g+cost))

    return None, created, [] # frontier exhausted, no dest

# ---------- BFS: BREADTH-FIRST SEARCH ----------
""" Frontier is a queue (FIFO), always expanding the shallowest node first so the path found has the fewest moves but not lowest cost.
    Neighbours are added in ascending order to satisfy tie-breaking. Goal is tested when a node is expanded.
    Tie-break: shallowest -> earliest added (queue order - already follows ascending-order rule).
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
    h = euclidean_heuristic(nodes, destinations) # Straight-line distance to the nearest destination

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

        g = record[2] # cost so far

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
    h = euclidean_heuristic(nodes, destinations) # Straight-line distance to the nearest destination

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

# ---------- CUS1: Bidirectional Flood fill ---------
"""Uses two BFS queues to search for a path, one from origin, one from goal. 
When the paths intersect, they are added into one path and exported as the output. Not guaranteed to be lowest cost or shortest"""

backward_trace = None # Hook to watch for backward expansion from the destination nodes

def cus1(nodes: dict, edges: dict, origin: int, destinations: list):
    goals = set(destinations)
    if origin in goals:
        return origin, 1, [origin]
    # 1st BFS search from origin
    root = (origin, None)
    created = 1 + len(goals) # Count the goals as well since it searches from there too
    visited1 = {origin: root}
    q = deque([root])
    # 2nd BFS search from goal
    q2 = deque()
    visited2 = {}
    for goal in goals:
        root2 = (goal, None)
        visited2[goal] = root2
        q2.append(root2)

    reversed_edges = {} # Reverse search requires the edges to be reversed
    for start, neighbours in edges.items():
        for end, cost in neighbours:
            reversed_edges.setdefault(end, []).append((start, cost))
    # Join path function
    def join_paths(node_1,node_2):
            path = list(path_from(node_1))
            meet_node = node_2[1] # Skips the meeting node so it isnt repeated
            while meet_node is not None:
                path.append(meet_node[0])
                meet_node = meet_node[1]  # Continues to the next node
            return path[-1], path

    # Main program loop
    while q and q2:
        fifo = q.popleft()  # FIFO - First in first out method to process the nodes
        curr_node = fifo[0] # the node currently being processed
        fifo2 = q2.popleft()
        curr_node2 = fifo2[0]
        for neighbour, cost in sorted_neighbours(edges, curr_node): # Checks all the nodes around the current node
            if neighbour in visited1:
                continue # Skip nodes that have already been discovered this search
            created += 1
            new_node = (neighbour, fifo) 
            visited1[neighbour] = new_node
            if neighbour in visited2:
                found, path = join_paths(new_node, visited2[neighbour])
                return found, created, path # stop if overlapping with second search. and return result
            q.append(new_node) # the loop will add new nodes to the queue if they haven't been discovered
        if backward_trace is not None: # Allows GUI to animate backward trace from the destination node
            backward_trace(curr_node2, [n for n, _ in sorted_neighbours(reversed_edges, curr_node2)])
        for neighbour, cost in sorted_neighbours(reversed_edges, curr_node2): # reversed edges is needed to traverse the graph
            if neighbour in visited2:
                continue 
            created += 1 
            new_node2 = (neighbour, fifo2)
            visited2[neighbour] = new_node2
            if neighbour in visited1:
                found, path = join_paths(visited1[neighbour], new_node2)
                return found, created, path # Stops when paths meet, both paths will be joined together to create the final output

            q2.append(new_node2)  
    return None, created, []

# ---------- CUS2: Modified unit-cost IDA* (Iterative Deepening A*) ----------
""" Works by setting a limit on f(n), the estimated num of moves where f(n) = g(n) + h(n). If a goal is not found within the limit set, raise limit and search again.
    g = number of moves so far, h = estimated number of moves remaining (straight-line distance / longest edge)
    Heuristic: Estimates number of remaining moves. Measures straight line distance from node to nearest destination then divide by longest edge length and round up.
               Number returned is the estimated number of moves remaining.
    Tie-breaking: fewest moves -> lowest cost -> smallest node number"""

def cus2(nodes: dict, edges: dict, origin: int, destinations: list):
    goals = set(destinations)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), len(nodes) + 100)) # Setting limit for recursion (path cannot be longer than no of nodes)

    goal_points = [nodes[g] for g in destinations if g in nodes]  # Coordinates (x, y) of every destination

    # Longest edge by straight-line distance between its nodes (not cost, h estimates moves). No move can cover more than this
    longest_edge = 0.0 
    for start, neighbours in edges.items():
        for end, _cost in neighbours:
            if start in nodes and end in nodes:
                longest_edge = max(longest_edge, math.dist(nodes[start], nodes[end]))

    # Heuristic: estimated moves left = distance to nearest goal / longest edge length (rounded up)
    def h(node: int) -> int:
        if node in nodes and goal_points and longest_edge > 0:
            nearest = min(math.dist(nodes[node], point) for point in goal_points) # Straight-line distance to the nearest goal/destination
            return math.ceil(nearest / longest_edge - 1e-9) # Gets estimated number of moves needed (- 1e-9 prevents float rounding errors)
        return 0 # No coordinates or edge length to estimate, h = 0 to prevent overestimating

    # Shared search state across iterations
    path = [origin] # The branch being explored right now
    on_path = {origin} # Check if the node is in current path
    created = 0 # Total search-tree nodes created across all iterations (cumulative)
    best_cost = math.inf # Cost of cheapest path so far (inf so any number will be smaller)
    best_path = None # No path until a goal is reached

    def limited_dfs(g: int, cost_so_far: float, limit: int):
        """ Depth-first search from the last node of 'path' while not going past f = limit, returns the smallest f that was cut off (math.inf if nothing was cut off)
            g - moves made so far, 'cost_so_far' is the total accumulated cost
            Goals reached are stored in best_cost/best_path """
        nonlocal created, best_cost, best_path # Ensures the variable are updated rather than creating new ones
        node = path[-1] # Current node is last one on path
        f = g + h(node) # Estimate = moves-so-far + estimated moves-to-go

        # Cut off if f is over the limit, otherwise check if this node is a goal
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
            if neighbour in on_path: # Skips nodes already on this branch (prevents cycles)
                continue

            new_cost = round(cost_so_far + cost, 9)
            if new_cost >= best_cost: # Skip if this branch already costs at least the same as the cheapest goal path found
                continue

            # Creates new search tree node
            created += 1
            path.append(neighbour)
            on_path.add(neighbour)

            smallest_cut = min(smallest_cut, limited_dfs(g + 1, new_cost, limit)) # Searches one move deeper, keeping the smallest f that was cut off

            # Backtracking - remove this neighbour so the next one can be tried
            path.pop()
            on_path.remove(neighbour)

        return smallest_cut # Smallest f cut off below this node (math.inf if nothing was cut off)

    # Iterations
    limit = h(origin) # First limit will be h(origin)
    while limit != math.inf: # Keep going while something was cut off last time (infinity means nothing was cut, so no solution)
        created += 1  # Rebuilds search tree on every iteration, restarts search with 0 moves
        next_limit = limited_dfs(0, 0, limit)
        if best_path is not None: # A goal was reached so no route with fewer moves exists, the cheapest path with this number of moves is kept
            return best_path[-1], created, best_path
        limit = next_limit # Raise budget to the smallest f value rejected

    return None, created, [] # Usually skipped if there is a solution to the test case, will only reach here if NO SOLUTIONS are available

# ---------- Code for Research: Find ALL Destinations ----------

# Wraps a search so it visits every destination, going to whichever one the search finds first
def visit_all(algorithm):
    def run(nodes, edges, origin, destinations): # Replacement search that repeats the search from wherever the last destination was reached
        remaining, path, created, goal = set(destinations), [origin], 0, None
        while remaining: # Keeps algorithm running until all destinations are visited
            found, count, segment = algorithm(nodes, edges, origin, sorted(remaining)) # Runs the search from current origin
            created += count
            if found is None: # None of the remaining destinations can be reached from here
                break
            path += segment[1:] # Segment starts at the old origin so skip it
            remaining.discard(found) # Removes found goal from the set of remaining destinations
            origin = goal = found # Update the goal reached to be new origin
        return goal, created, (path if goal is not None else [])
    return run

# ---------- METHODS DICTIONARY ----------
methods: dict = { # Maps each method name to its function
    'DFS': dfs,
    'BFS': bfs,
    'GBFS': gbfs,
    'AS': a_star,
    'CUS1': cus1,
    'CUS2': cus2
}