import sys
import math
import tkinter as tk
from tkinter import font as tkfont

from algorithms import methods, visit_all
from fileparser import parse_file
from search import route_cost, run_measured

BG = '#ffffff'
R = 18
SIDE_SHIFT = 5
LABEL_SHIFT = 12 # Weight label is 12px away from edge line
LABEL_POS = 0.3 # Weight label are located 30% of the edge length

FOOTER = 150 # Space kept at the bottom of the window for the legend and status text
LEGEND_Y = 130 # Center of legend row
LEGEND_ICON = 16 # Diameter of legend icon
STATUS_Y = 10 # Baseline of status text

# Colours for each state of the search, and the outline that shows a node's role
UNEXPLORED, FRONTIER, EXPLORED, CURRENT, FINAL, FINAL_EDGE = '#cccccc', "#ff8400", "#8cd1e6", "#ff0000", "#45af4a", "#297232"
ORIGIN_OUTLINE, GOAL_OUTLINE, NORMAL_OUTLINE = "#0044ff", "#ff0000", '#555555'
LEGEND = [('Unexplored', UNEXPLORED, NORMAL_OUTLINE), # White with black outline
          ('Frontier', FRONTIER, NORMAL_OUTLINE), # Orange with black outline
          ('Explored', EXPLORED, NORMAL_OUTLINE), # Light blue with black outline
          ('Currently exploring', CURRENT, NORMAL_OUTLINE), # Red with black outline
          ('Final path', FINAL, NORMAL_OUTLINE), # Green with black outline
          ('Origin', UNEXPLORED, ORIGIN_OUTLINE), # White with light blue outline
          ('Destination', UNEXPLORED, GOAL_OUTLINE)] # White with red outline

def node_positions(nodes, width, height): # Mapping node coordinates to the display
    xs = [x for x, _ in nodes.values()]
    ys = [y for _, y in nodes.values()]

    # Get size of graph needed in terms of weight unit
    graph_w = max(xs) - min(xs)
    graph_h = max(ys) - min(ys)
    pad = 3 * R

    # Scale so the graph fits the window, then centre it and map nodes to their positions
    scale = min((width - 2 * pad) / (graph_w or 1), (height - 2 * pad) / (graph_h or 1))
    left = (width - graph_w * scale) / 2
    bottom = (height + graph_h * scale) / 2
    return {node_id: (left + (x - min(xs)) * scale, bottom - (y - min(ys)) * scale)
            for node_id, (x, y) in nodes.items()}

def draw_edge(canvas, start, end, cost, two_way, colour, width):
    (x1, y1), (x2, y2) = start, end
    length = math.hypot(x2 - x1, y2 - y1) or 1 # Length of edge in pixels ("or 1" prevents dividing by zero)
    ux, uy = (x2 - x1) / length, (y2 - y1) / length # Calculates unit vector of edge length
    px, py = -uy, ux # Perpendicular unit vector (edge rotated by 90 degrees, used for two-way edges)
    shift = SIDE_SHIFT if two_way else 0 #Shifts two-way edge by a bit so it is not overlapping

    # Line starts 2px outside the start node and stops 4px outside the end node (so the arrowhead is visible)
    ax, ay = x1 + px * shift + ux * (R + 2), y1 + py * shift + uy * (R + 2)
    bx, by = x2 + px * shift - ux * (R + 4), y2 + py * shift - uy * (R + 4)
    canvas.create_line(ax, ay, bx, by, fill=colour, width=width, arrow=tk.LAST, arrowshape=(10, 12, 4))
    return ax + (bx - ax) * LABEL_POS + px * LABEL_SHIFT, ay + (by - ay) * LABEL_POS + py * LABEL_SHIFT, cost # Positioning of cost label + return calculated values

# Draws the cost label on a white box to prevent going missing when overlapped
def draw_label(canvas, x, y, cost):
    number = int(cost) if cost == int(cost) else cost # Display cost as whole number if not decimal
    text = canvas.create_text(x, y, text=str(number), font=('Arial', 8))
    canvas.tag_lower(canvas.create_rectangle(canvas.bbox(text), fill=BG, outline=''), text)

def main():
    if len(sys.argv) not in (3, 4) or (len(sys.argv) == 4 and sys.argv[3].lower() != '--all'): # Validates 3 arguments or 4 with the optional --all toggle
        print('Usage: python gui.py <filename> <method> [--all]')
        print('  --all: visits ALL destinations instead of stopping after one')
        print(f"Methods available: {', '.join(methods)}")
        sys.exit(1)

    filename = sys.argv[1]
    method = sys.argv[2].upper()
    visit_every = len(sys.argv) == 4 # True when --all is given

    if method not in methods: # Error if method is incorrect
        print(f"Error: '{method}' does not exist!")
        print(f"Methods available: {', '.join(methods)}")
        sys.exit(1)

    try: # Error handling for file parsing
        nodes, edges, origin, destinations = parse_file(filename)
    except Exception as exc:
        print(f"Error while reading '{filename}': {exc}")
        sys.exit(1)
    goals = set(destinations)

    # Gets all the information by running it once
    algorithm = visit_all(methods[method]) if visit_every else methods[method] # Wraps the method if all destinations must be visited
    goal, created, path, time_taken, memory_text = run_measured(algorithm, nodes, edges, origin, destinations)

    # ---------- RECORDING THE SEARCH FOR GUI PLAYBACK ----------
    events = [] # Holds an entry for each expanded node: (leg number, node, neighbours)
    legs = [] # Start node of each leg (usually one leg or one per destination reached when --all is used)
    class WatchedEdges(dict): # Watches over the expanded edges to record expansion
        def __getitem__(self, node):
            neighbours = super().__getitem__(node) # Looks up the dictionary record
            events.append((len(legs) - 1, node, [n for n, _ in neighbours])) # Records the event under the current leg
            return neighbours

        # Overrides get to use the code above instead of using normal implementation
        def get(self, node, default = None): 
            return self[node] if node in self else default

    # Notes down where each leg starts so the playback knows when a new leg begins
    def mark_legs(search):
        def run(nodes, edges, start, dests):
            legs.append(start) # Events recorded from here on belong to this leg
            return search(nodes, edges, start, dests)
        return run

    # Runs the search algorithm
    recorder = mark_legs(methods[method])
    if visit_every:
        recorder = visit_all(recorder) # Each leg is marked, then the goal reached becomes the next origin
    recorder(nodes, WatchedEdges(edges), origin, destinations)
    path_edges = set(zip(path, path[1:])) # Edges used by the final path

    performance = f'Time taken: {time_taken:.3f} ms   |   Memory usage (peak): {memory_text}' # Performance metrics for the search

    # Forms the text for output when destination is found/not found
    if goal is None:
        result = f'No destination can be reached. Nodes created: {created}'
    else:
        cost = route_cost(edges, path)
        route = ' -> '.join(str(n) for n in path)
        result = result = (f'Goal reached: Node {goal}   |   Nodes created: {created}   |   Number of moves: {len(path) - 1}   |   Total cost: {cost:g} \nFinal route: {route}')
        if visit_every:
            order = []
            for n in path: # Goals in the order they were first reached
                if n in goals and n not in order:
                    order.append(n)
            missed = [d for d in dict.fromkeys(destinations) if d not in order] # Any unreachable destination will be added here
            result += f"\nGoals visited (in order): {' -> '.join(map(str, order))}"
            if missed:
                result += f"\nGoals NOT reachable: {', '.join(map(str, missed))}"

    # Works out which node gets the origin (blue) outline and which get the destination (red) outline while a given leg is searched
    def node_role(leg):
        if not visit_every: # Normal mode keeps the original origin and every destination outlined
            return origin, goals
        reached = set(legs[1:leg + 1]) # Every leg after the first starts at the goal reached by the leg before it
        return legs[leg], goals - reached # Current origin is blue, destinations not yet reached stay red, reached ones go back to normal

    # Animating the search step by step
    def frames():
        explored, frontier, iteration, current_leg = set(), set(), 1, None
        yield explored, frontier, None, f'{method}: {len(events)} expansions to show...', False, node_role(0)
        for step, (leg, node, neighbours) in enumerate(events, 1):  # Loops over all expansions and numbers them
            if leg != current_leg: # A new leg has started (new origin), so begin again with a clean graph
                explored.clear()
                frontier.clear()
                iteration = 1
                current_leg = leg
            elif node == legs[leg] and node in explored: # Resets the graph if the search gets restarted (used for CUS2/IDA* search)
                explored.clear()
                frontier.clear()
                iteration += 1
            explored.add(node)
            frontier.update(neighbours) # Sets neighbour nodes as frontier
            leg_text = f'Leg {leg + 1} of {len(legs)} (from node {legs[leg]}) - ' if visit_every else '' # Only shown when --all is used
            roles = node_role(leg) # Origin and destination to outline during this leg

            # Yield used as an alternative for return, remembers previous state before pausing (important to keep track). Also has iteration for IDA* search
            yield explored, frontier, node, leg_text + f'Step {step} of {len(events)}: exploring node {node}' + (f' (iteration {iteration})' if iteration > 1 else ''), False, roles
        yield explored, frontier, None, 'SEARCH COMPLETED! \n\n' + performance + '\n' + result, True, (origin, goals) # Once finished, outline the real origin and every destination again

    def draw(frame):
        explored, frontier, current, message, finished, (leg_origin, leg_goals) = frame # Unpacks information as a frame
        canvas.delete('all') # Clear canvas before drawing graph

        # Decide canvas size, set node position and edges
        width, height = canvas.winfo_width(), canvas.winfo_height()
        pos = node_positions(nodes, width, height - FOOTER)
        pairs = {(a, b) for a, numbers in edges.items() for b, _ in numbers}

        # Looping through all edges
        labels = []
        for a, numbers in edges.items():
            for b, cost in numbers:
                if a == b or a not in pos or b not in pos: # Skips self-loops and undefined edges
                    continue
                on_path = finished and (a, b) in path_edges
                labels.append(draw_edge(canvas, pos[a], pos[b], cost, (b, a) in pairs, FINAL_EDGE if on_path else '#000000', 3.5 if on_path else 1.5)) # Highlights the final edge when everything is finished
        for x, y, cost in labels:
            draw_label(canvas, x, y, cost)

        # Determines fill colour according to node type
        for node, (x, y) in pos.items():
            if finished and node in path:
                fill = FINAL
            elif node == current:
                fill = CURRENT
            elif node in explored:
                fill = EXPLORED
            elif node in frontier:
                fill = FRONTIER
            else:
                fill = UNEXPLORED

            # Setting the outline for nodes
            if node == leg_origin: # Origin of the leg being searched (the real origin when --all is off)
                outline = ORIGIN_OUTLINE
                outline_width = 4
            elif node in leg_goals: # Destination of the leg being searched
                outline = GOAL_OUTLINE
                outline_width = 4
            else:
                outline = NORMAL_OUTLINE
                outline_width = 1.5

            # Draws the node
            canvas.create_oval(x - R, y - R, x + R, y + R, fill=fill, outline=outline, width=outline_width)
            canvas.create_text(x, y, text=str(node), font=('Arial', 10, 'bold'))

        # Legend icon and text
        x = 20
        icon_top = height - LEGEND_Y - LEGEND_ICON / 2 # Top edge of legend icon
        icon_bottom = height - LEGEND_Y + LEGEND_ICON / 2 # Bottom edge of legend icon
        for text, fill, outline in LEGEND:
            canvas.create_oval(x, icon_top, x + LEGEND_ICON, icon_bottom, fill=fill, outline=outline, width=1.5 if outline == NORMAL_OUTLINE else 3)
            canvas.create_text(x + LEGEND_ICON + 6, height - LEGEND_Y, text=text, anchor='w', font=legend_font) # Label of legend
            x += 28 + LEGEND_ICON + legend_font.measure(text) # Positions the next legend right beside the previous one

        # Status text anchored to bottom so it grows upwards
        canvas.create_text(20, height - STATUS_Y, text=message, anchor='sw', width=width - 40, font=('Arial', 10, 'bold' if not finished else 'normal'))

    # Creates graph window
    root = tk.Tk()
    root.title(f'{filename} - {method}' + (' (all destinations)' if visit_every else ''))
    root.geometry('1024x768')
    root.minsize(760, 520)
    legend_font = tkfont.Font(family='Arial', size=9)
    canvas = tk.Canvas(root, bg=BG, highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    shown = [] #Current frame on screen, stored so it can be redrawn on resize
    next_frame = frames()
    delay = max(20, min(700, 15000 // max(1, len(events)))) # Adjust delay (20-700 ms) according to search complexity, caps at 15 seconds

    # Animates the next frame of the search
    def animate():
        frame = next(next_frame, None)
        if frame is not None:
            shown[:] = [frame]
            draw(frame)
            root.after(delay, animate)

    canvas.bind('<Configure>', lambda _e: shown and draw(shown[0]))
    root.after(500, animate) # Delay to ensure window has size set correctly first
    root.mainloop()

if __name__ == '__main__':
    main()