# CURRENTLY USED FOR TESTING, ONLY RENDERS IN NODES + EDGES, FOR FULL GUI IT WILL SHOW PATH TAKEN + ORIGIN AND DESTINATIONS

import sys
import math
import tkinter as tk
from fileparser import parse_file

BG = '#ffffff'
R = 18
SIDE_SHIFT = 5
LABEL_SHIFT = 12 # Weight label is 12px away from edge line
LABEL_POS = 0.3 # Weight label are located 30% of the edge length


def node_positions(nodes, width, height):
    xs = [x for x, _ in nodes.values()]
    ys = [y for _, y in nodes.values()]

    # Get size of graph needed in terms of weight unit
    graph_w = max(xs) - min(xs)
    graph_h = max(ys) - min(ys)

    pad = R * 3

    # Determining the scale to be used for rendering
    scale = min((width - 2 * pad) / (graph_w or 1),
                (height - 2 * pad) / (graph_h or 1))

    # Positioning of left-most and bottom-most edges so the graph is centered
    left = (width - graph_w * scale) / 2
    bottom = (height + graph_h * scale) / 2

    # Maps the node to their respective positions
    return {node_id: (left + (x - min(xs)) * scale, bottom - (y - min(ys)) * scale)
            for node_id, (x, y) in nodes.items()}

def draw_edge(canvas, start, end, cost, two_way): # Draws the graph edges
    x1, y1 = start
    x2, y2 = end

    length = math.hypot(x2 - x1, y2 - y1) or 1 # Length of the edge in pixels ("or 1" avoids dividing by zero)
    ux, uy = (x2 - x1) / length, (y2 - y1) / length # Calculates unit vector of the edge length
    px, py = -uy, ux # Perpendicular unit vector (edge rotated by 90 degrees, used for two-way edges)
    shift = SIDE_SHIFT if two_way else 0 # Shifts two-way edge by a bit so it is not overlapping

    # Setting the edge line so it is 2px outside of start node and 4px outside for end node (allow arrow to be visible)
    line_x1 = x1 + px * shift + ux * (R + 2)
    line_y1 = y1 + py * shift + uy * (R + 2)
    line_x2 = x2 + px * shift - ux * (R + 4)
    line_y2 = y2 + py * shift - uy * (R + 4)

    # Draws the edge line, ends with an arrow
    canvas.create_line(line_x1, line_y1, line_x2, line_y2, fill='#000000', width=1.5, arrow=tk.LAST, arrowshape=(10, 12, 4))

    # Positions cost label at 30% of the edge length and shifts it 12px away from edge line
    label_x = line_x1 + (line_x2 - line_x1) * LABEL_POS + px * LABEL_SHIFT
    label_y = line_y1 + (line_y2 - line_y1) * LABEL_POS + py * LABEL_SHIFT
    return label_x, label_y, cost

# Draws the cost label on a white box to prevent going missing when overlapped
def draw_label(canvas, x, y, cost):
    number = int(cost) if cost == int(cost) else cost # Display cost as whole number if not decimal
    text = canvas.create_text(x, y, text=str(number), fill='#000000', font=('Arial', 8))

    left, top, right, bottom = canvas.bbox(text)
    box = canvas.create_rectangle(left, top, right, bottom, fill=BG, outline='')
    canvas.tag_lower(box, text)

# Draws the node
def draw_node(canvas, node_id, position):
    x, y = position
    canvas.create_oval(x - R, y - R, x + R, y + R, fill='#dddddd', outline='#555555', width=1.5)
    canvas.create_text(x, y, text=str(node_id), font=('Arial', 10, 'bold'))

# Draws entire graph
def draw(canvas, nodes, edges):
    canvas.delete('all') # Clear canvas before drawing graph

    # Current canvas size in pixels (900px and 600px set as fallback value)
    width = canvas.winfo_width() or 900
    height = canvas.winfo_height() or 600
    pos = node_positions(nodes, width, height) # Decides node positions

    # Defines edges as a 'from' and 'to' pair.
    all_pairs = {(frm, to) for frm, numbers in edges.items() for to, _ in numbers}

    # Creates the edge
    labels = []
    for frm, numbers in edges.items():
        for to, cost in numbers:
            labels.append(draw_edge(canvas, pos[frm], pos[to], cost, two_way=(to, frm) in all_pairs))

    # Creates cost label for the lines
    for x, y, cost in labels:
        draw_label(canvas, x, y, cost)

    # Draw nodes after so it is placed on top of the edge
    for node_id, position in pos.items():
        draw_node(canvas, node_id, position)

if __name__ == '__main__':
    if len(sys.argv) != 2: # Requires one argument only atm, just filename as this is just to visualise graph
        print('Usage: python visualise.py <filename>')
        sys.exit(1)

    # Parsing the input file, only accept nodes and edges atm since its testing
    nodes, edges, *_ = parse_file(sys.argv[1])

    # Creates graph window
    root = tk.Tk()
    root.title(sys.argv[1])
    root.geometry('900x650')
    canvas = tk.Canvas(root, bg=BG, highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)
    canvas.bind('<Configure>', lambda _e: draw(canvas, nodes, edges)) # Redraw graph to fit when window is resized
    root.mainloop()