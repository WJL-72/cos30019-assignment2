def parse_file(filename: str): # Function to nom nom information from yummy file I AM LOSING MY MIND

    # Defining variables for the file reader
    nodes = {} # { node: (x, y), ...}
    edges = {} # {node: [neighbour, cost], ...}
    origin = None
    destinations = [] # Accepts list of destinations
    section = None

    with open(filename, 'r') as file:
        for lineno, raw in enumerate(file, start=1):
            line = raw.strip() # Strips whitespaces and line breaks
            if not line:
                continue

            # Gets header of current section
            if line == 'Nodes:':
                section = 'nodes'
            elif line == 'Edges:':
                section = 'edges'
            elif line == 'Origin:':
                section = 'origin'
            elif line == 'Destinations:':
                section = 'destinations'

            # Code to parse data based on section
            elif section == 'nodes':

                node_id, coords = line.split(':', 1) # Splits on first colon (e.g. "1: (4, 1)")
                node_id = int(node_id.strip())

                x, y = map(int, coords.strip().strip('()').split(',')) # Obtains x and y coordinates from the rest of the text
                nodes[node_id] = (x, y) # Stores node ID (number) and coords
                edges.setdefault(node_id, []) # Assigns empty list as default edges

            elif section == 'edges':
                edge_text, cost_text = line.rsplit(':', 1) # Edge text and cost text are unprocessed versions
                from_node, to_node = map(int, edge_text.strip().strip('()').split(',')) # Gets the source and destination node of an edge
                cost = float(cost_text.strip()) # Float to support decimal costs

                edges.setdefault(from_node, []) # Fallback in case source node is NOT part of node_id
                edges[from_node].append((to_node, cost)) # Assigns destination node and cost to source node

            elif section == 'origin':
                origin = int(line.strip())

            elif section == 'destinations':
                destinations = [int(dest.strip()) for dest in line.split(';') if dest.strip()] # Reads destination nodes while skipping trailing semicolon

    return nodes, edges, origin, destinations