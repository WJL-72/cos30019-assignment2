import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Gets base directory of the files
TEST_DIR = os.path.join(BASE_DIR, 'test_cases') # Sets the test case file directory

def resolve_path(filename: str):
    for folder in ('', TEST_DIR):
        for name in (filename, filename + '.txt'): # Fail-safe mechanism, works even if user doesn't include file extension (if they input filename only)
            candidate = os.path.join(folder, name)
            if os.path.isfile(candidate): # Checks if the file name exists, if it does return it
                return candidate
    return filename # Only reaches here if no matching file is found, used for error handling

def to_number(text: str):
    value = float(text.strip()) # Accepts decimals as well as whole numbers
    return int(value) if value == int(value) else value # Whole numbers stay as int instead of decimal

def parse_file(filename: str):
    # Defining variables for the file reader
    nodes = {} # { node: (x, y), ...}
    edges = {} # { node: [(neighbour, cost), ...] }
    origin = None
    destinations = [] # Accepts list of destinations
    section = None
    headers = {'nodes:': 'nodes', 'edges:': 'edges', 'origin:': 'origin', 'destinations:': 'destinations'} # Section keywords

    with open(resolve_path(filename), 'r') as file: # Resolves the path before opening
        for lineno, raw in enumerate(file, start=1):
            line = raw.strip() # Strips whitespaces and line breaks
            if not line:
                continue
            try:
                # Gets header of current section
                if line.lower() in headers:
                    section = headers[line.lower()]

                # Code to parse data based on section
                elif section == 'nodes':
                    node_id, coords = line.split(':', 1) # Splits on first colon (e.g. "1: (4, 1)")
                    node_id = int(node_id.strip())

                    x, y = map(to_number, coords.strip().strip('()').split(',')) # Obtains x and y coordinates from the rest of the text
                    nodes[node_id] = (x, y) # Stores node ID (number) and coords
                    edges.setdefault(node_id, []) # Assigns empty list as default edges

                elif section == 'edges':
                    edge_text, cost_text = line.rsplit(':', 1) # Edge text and cost text are unprocessed versions
                    from_node, to_node = map(int, edge_text.strip().strip('()').split(',')) # Gets the source and destination node of an edge
                    cost = float(cost_text.strip()) # Float to support decimal costs

                    edges.setdefault(from_node, []) # Fallback in case source node is not listed under Nodes
                    edges[from_node].append((to_node, cost)) # Assigns destination node and cost to source node

                elif section == 'origin':
                    if origin is not None: # Checking if multiple origins are given
                        raise ValueError('more than one origin given')
                    origin = int(line)

                elif section == 'destinations':
                    destinations.extend(int(dest) for dest in line.split(';') if dest.strip()) # Reads destination nodes while skipping trailing semicolon

                else: # Reports if there is content found before a section
                    raise ValueError("data found before a section keyword ('Nodes:', 'Edges:', 'Origin:' or 'Destinations:')")

            except ValueError as exc: # Reports which line is wrong instead of a vague error
                raise ValueError(f"line {lineno} ('{line}'): {exc}") from None

    # Checks the problem file is complete before any search starts
    if origin is None:
        raise ValueError("no origin given (missing 'Origin:' section)")
    if not destinations:
        raise ValueError("no destinations given (missing 'Destinations:' section)")
    if origin not in nodes:
        raise ValueError(f"origin {origin} is not listed under 'Nodes:'")
    
    unknown = [d for d in dict.fromkeys(destinations) if d not in nodes]
    if unknown:
        raise ValueError(f"destination(s) {', '.join(map(str, unknown))} not listed under 'Nodes:'")

    return nodes, edges, origin, destinations