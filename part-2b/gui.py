"""
gui.py - GUI for the Traffic-Based Route Guidance System (TBRGS), Assignment 2B

Pick an origin, destination, departure time and a few settings, and it shows up to 5 routes
with their travel times on a map of the Boroondara sites. Defaults come from config.toml

Run:    python gui.py
The GUI only collects the input and draws the result. All the work is done by the same code as main.py:
    calculate_time.build_graph() - predicts the traffic and gives every road a travel time
    k_routes.k_shortest_paths_via() - finds the routes with the chosen 2A search method
If the processed data or the chosen model is missing, it offers to build it first (same as main.py does).
"""
import os
import tkinter as tk
import tomllib
from datetime import datetime
from tkinter import messagebox, ttk

import pandas as pd

import algorithms
import calculate_time as backend
import k_routes
import main as pipeline # only used to build missing data/models

HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_NAMES = {"lstm": "LSTM", "gru": "GRU", "rf": "Random Forest"} # text shown in the drop-down
MAX_ROUTES = 5
ROUTE_COLOURS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"] # Okabe-Ito colours, best route first
ROAD_GREY, GUESS_GREY, NODE_FILL, NODE_OUTLINE = "#c8c8c8", "#e3e3e3", "#ffffff", "#555555"
ORIGIN_OUTLINE, DEST_OUTLINE = "#0044ff", "#ff0000" # blue and red like the 2A GUI
NODE_R = 8


def load_config() -> dict:
    with open(os.path.join(HERE, "config.toml"), "rb") as file:
        return tomllib.load(file)["gui"]


class App:
    def __init__(self, root: tk.Tk, cfg: dict):
        self.root, self.cfg = root, cfg
        self.sites = pd.read_csv(backend.SITES_FILE, index_col=0) # x, y and road names of every site
        self.edges = pd.read_csv(backend.EDGES_FILE)
        self.road_of = {frozenset((a, b)): road for a, b, road in zip(self.edges["site_a"], self.edges["site_b"], self.edges["road"])}

        # Text for the drop-downs, e.g. "2000  BURWOOD_HWY | TOORAK_RD | WARRIGAL_RD"
        self.site_text = {s: f"{s}  {row['roads']}" for s, row in self.sites.iterrows()}
        self.model_key = {MODEL_NAMES.get(k, k.upper()): k for k in backend.MODEL_FILES}
        self.date_text = [f"{d:%Y-%m-%d} ({d:%a})" for d in pd.date_range(cfg["first_date"], cfg["last_date"])]
        self.time_text = [f"{m // 60:02d}:{m % 60:02d}" for m in range(0, 24 * 60, backend.INTERVAL_MIN)]

        self.routes, self.legs, self.notes = [], {}, [] # result of the last search
        self.origin, self.dest, self.date, self.time = (tk.StringVar() for _ in range(4))
        self.model, self.method, self.flow_mode, self.count = (tk.StringVar() for _ in range(4))
        self.guessed = tk.BooleanVar()
        self.build_widgets()
        self.apply_defaults()

    # ---------- LAYOUT ----------
    def build_widgets(self):
        self.root.title("TBRGS - Traffic-Based Route Guidance System")
        self.root.geometry("1150x740")
        self.root.minsize(950, 680)

        left = ttk.Frame(self.root, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)
        self.canvas = tk.Canvas(self.root, bg="white", highlightthickness=0)
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _e: self.draw())

        # Trip input
        trip = ttk.LabelFrame(left, text="Trip", padding=8)
        trip.pack(fill=tk.X)
        sites = list(self.site_text.values())
        origin_box = self.add_row(trip, "Origin", lambda row: ttk.Combobox(row, textvariable=self.origin, state="readonly", values=sites, width=34))
        dest_box = self.add_row(trip, "Destination", lambda row: ttk.Combobox(row, textvariable=self.dest, state="readonly", values=sites, width=34))
        for box in (origin_box, dest_box): # new trip = old routes are cleared, so the map is never out of date
            box.bind("<<ComboboxSelected>>", lambda _e: self.clear_routes())
        self.add_row(trip, "Date (YYYY-MM-DD)", lambda row: ttk.Combobox(row, textvariable=self.date, values=self.date_text, width=34)) # not readonly, other dates can be typed
        self.add_row(trip, "Departure time", lambda row: ttk.Combobox(row, textvariable=self.time, state="readonly", values=self.time_text, width=34))

        # Parameter settings
        settings = ttk.LabelFrame(left, text="Settings", padding=8)
        settings.pack(fill=tk.X, pady=8)
        self.add_row(settings, "ML model", lambda row: ttk.Combobox(row, textvariable=self.model, state="readonly", values=list(self.model_key), width=34))
        self.add_row(settings, "Search method", lambda row: ttk.Combobox(row, textvariable=self.method, state="readonly", values=list(algorithms.methods), width=34))

        def flow_buttons(row): # two flow options side by side
            box = ttk.Frame(row)
            ttk.Radiobutton(box, text="Average per approach", variable=self.flow_mode, value="average").pack(side=tk.LEFT)
            ttk.Radiobutton(box, text="Sum of approaches", variable=self.flow_mode, value="sum").pack(side=tk.LEFT, padx=6)
            return box
        self.add_row(settings, "Flow used", flow_buttons)
        self.add_row(settings, "Guessed roads", lambda row: ttk.Checkbutton(row, text="Include guessed edges", variable=self.guessed))
        self.add_row(settings, "Routes to show (1-5)", lambda row: ttk.Spinbox(row, from_=1, to=MAX_ROUTES, textvariable=self.count, width=5))

        buttons = ttk.Frame(left)
        buttons.pack(fill=tk.X)
        ttk.Button(buttons, text="Find routes", command=self.find).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Reset to defaults", command=self.apply_defaults).pack(side=tk.LEFT, padx=8)

        self.status = ttk.Label(left, wraplength=440, justify=tk.LEFT)
        self.status.pack(fill=tk.X, pady=8)

        # Route list (rows have the same colour as the route on the map) and details of the selected route
        self.tree = ttk.Treeview(left, columns=("route", "time", "dist", "stops"), show="headings", height=5, selectmode="browse")
        for column, heading, width in (("route", "Route", 55), ("time", "Time (min)", 95), ("dist", "Distance (km)", 105), ("stops", "Sites", 55)):
            self.tree.heading(column, text=heading)
            self.tree.column(column, width=width, anchor=tk.CENTER)
        for i, colour in enumerate(ROUTE_COLOURS):
            self.tree.tag_configure(f"r{i}", foreground=colour)
        self.tree.pack(fill=tk.X)
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self.show_selected())

        holder = ttk.Frame(left)
        holder.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.detail = tk.Text(holder, height=10, width=67, state=tk.DISABLED, font=("Consolas", 9), wrap=tk.NONE)
        scrollbar = ttk.Scrollbar(holder, command=self.detail.yview)
        self.detail.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.detail.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    @staticmethod
    def add_row(parent, label: str, make_widget):
        # label + widget on one line. The widget is made INSIDE the row, else the row hides it
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=19).pack(side=tk.LEFT)
        widget = make_widget(row)
        widget.pack(side=tk.LEFT)
        return widget

    # ---------- ACTIONS ----------
    def apply_defaults(self):
        c = self.cfg
        self.origin.set(self.site_text[c["origin"]])
        self.dest.set(self.site_text[c["destination"]])
        self.date.set(f"{c['departure_date']} ({datetime.strptime(c['departure_date'], '%Y-%m-%d'):%a})")
        self.time.set(c["departure_time"])
        self.model.set(MODEL_NAMES.get(c["model"], c["model"].upper()))
        self.method.set(c["method"].upper())
        self.flow_mode.set(c["flow_mode"])
        self.guessed.set(c["include_guessed"])
        self.count.set(str(c["routes"]))
        self.clear_routes()

    def clear_routes(self):
        self.routes, self.legs, self.notes = [], {}, []
        self.tree.delete(*self.tree.get_children())
        self.set_detail("")
        self.set_status("Choose a trip and press 'Find routes'.")
        self.draw()

    def find(self):
        origin, dest = self.site_id(self.origin), self.site_id(self.dest)
        if origin == dest:
            return self.set_status("Origin and destination are the same site.", error=True)
        try:
            day = datetime.strptime((self.date.get().split() or [""])[0], "%Y-%m-%d") # drops the "(Mon)" part
        except ValueError:
            return self.set_status("The date must look like 2006-10-30 (YYYY-MM-DD).", error=True)
        try:
            count = int(self.count.get())
        except ValueError:
            return self.set_status("Routes to show must be a whole number from 1 to 5.", error=True)
        count = max(1, min(MAX_ROUTES, count))
        self.count.set(str(count))

        model, method = self.model_key[self.model.get()], self.method.get()
        if not self.prepare(model):
            return
        backend.FLOW_PER_APPROACH = self.flow_mode.get() == "average" # the backend reads these two when it builds the graph
        backend.INCLUDE_GUESSED_EDGES = self.guessed.get()

        self.set_status("Calculating... the first run loads the model, it can take a few seconds.")
        self.root.config(cursor="watch")
        self.root.update()
        try:
            graph = backend.build_graph(model, f"{day:%Y-%m-%d} {self.time.get()}", origin, [dest])
            found = k_routes.k_shortest_paths_via(graph["nodes"], graph["edges"], [origin, dest], method=method)
        except backend.TravelTimeError as exc: # problems the backend explains itself (missing file, bad date...)
            return self.set_status("\n".join(filter(None, [str(exc), exc.hint])), error=True)
        except ValueError as exc: # e.g. unknown site or method
            return self.set_status(str(exc), error=True)
        except Exception as exc: # anything else is shown instead of crashing the window
            return messagebox.showerror("TBRGS", f"The route calculation failed:\n{exc}")
        finally:
            self.root.config(cursor="")

        self.legs = {(r.from_site, r.to_site): r for r in graph["edge_table"].itertuples()}
        self.routes = [(cost, path, sum(self.legs[(a, b)].distance_km for a, b in zip(path, path[1:]))) for cost, path in found[:count]]
        self.notes = graph["notes"]
        self.tree.delete(*self.tree.get_children())
        for i, (cost, path, km) in enumerate(self.routes):
            self.tree.insert("", tk.END, iid=str(i), tags=(f"r{i}",), values=(i + 1, f"{cost:.2f}", f"{km:.2f}", len(path)))

        head = f"{self.model.get()} | {origin} -> {dest} | depart {graph['time']:%a %Y-%m-%d %H:%M} ({graph['mode']} traffic) | {method}"
        if self.routes:
            self.set_status(f"{head}\nFound {len(self.routes)} route(s).")
            self.tree.selection_set("0") # best route, also calls show_selected()
        else:
            cut_off = [s for s in (origin, dest) if not graph["edges"][s]] # sites with no road at all
            why = f" Site(s) {', '.join(map(str, cut_off))} only connect through guessed edges, tick 'Include guessed edges'." if cut_off else ""
            self.set_status(f"{head}\nNo route found.{why}", error=True)
            self.show_selected()

    def prepare(self, model: str) -> bool:
        # Builds the processed data and the model if they are missing (same as main.py), True if ready
        data_ready = all(p.exists() for _label, _script, _inputs, outputs in pipeline.DATA_STAGES for p in outputs)
        model_ready = pipeline.MODELS[model][2].exists()
        if data_ready and model_ready:
            return True
        what = "processed data" if not data_ready else f"{self.model.get()} model"
        if not messagebox.askyesno("TBRGS", f"The {what} has not been built yet.\nBuild it now? This can take a few minutes and the window will freeze (progress shows in the terminal)."):
            self.set_status(f"The {what} is missing, run 'python main.py {model} \"<time>\" <origin> <destination>' to build it.", error=True)
            return False
        self.set_status("Building, see the terminal for progress...")
        self.root.config(cursor="watch")
        self.root.update()
        try:
            pipeline.ensure_ready(model)
        except pipeline.PipelineError as exc:
            self.set_status(f"Build failed: {exc}", error=True)
            return False
        finally:
            self.root.config(cursor="")
        return True

    def site_id(self, var: tk.StringVar) -> int:
        return int(var.get().split()[0])

    def selected(self):
        chosen = self.tree.selection()
        return int(chosen[0]) if chosen else None

    def show_selected(self):
        index = self.selected()
        lines = []
        if index is not None:
            cost, path, km = self.routes[index]
            lines = [f"Route {index + 1}: {cost:.2f} min, {km:.2f} km", ""]
            for a, b in zip(path, path[1:]):
                leg = self.legs[(a, b)]
                road = self.road_of.get(frozenset((a, b)), "")[:14]
                flow = "  n/a" if pd.isna(leg.flow_veh_per_h) else f"{leg.flow_veh_per_h:5.0f}" # n/a = no prediction, 60 km/h used
                lines.append(f"{a:>4} -> {b:<4} {road:<14} {leg.distance_km:.2f} km {flow} veh/h {leg.speed_kmh:3.0f} km/h {leg.travel_time_min:.1f} min")
            lines += ["", "Each road includes a 30 s intersection delay."]
        if self.notes:
            lines += ["", "Notes:"] + [f"- {note}" for note in self.notes]
        self.set_detail("\n".join(lines))
        self.draw()

    def set_status(self, text: str, error: bool = False):
        self.status.config(text=text, foreground="#c00000" if error else "black")

    def set_detail(self, text: str):
        self.detail.config(state=tk.NORMAL)
        self.detail.delete("1.0", tk.END)
        self.detail.insert("1.0", text)
        self.detail.config(state=tk.DISABLED)

    # ---------- MAP ----------
    def positions(self, width: int, height: int) -> dict:
        # fits the site coordinates (km) into the canvas, same idea as node_positions() in the 2A GUI
        xs, ys = self.sites["x"], self.sites["y"]
        pad = 50
        scale = min((width - 2 * pad) / ((xs.max() - xs.min()) or 1), (height - 2 * pad - 30) / ((ys.max() - ys.min()) or 1))
        left = (width - (xs.max() - xs.min()) * scale) / 2
        bottom = (height - 30 + (ys.max() - ys.min()) * scale) / 2
        return {s: (left + (row["x"] - xs.min()) * scale, bottom - (row["y"] - ys.min()) * scale) for s, row in self.sites.iterrows()}

    def draw(self):
        c = self.canvas
        c.delete("all")
        width, height = c.winfo_width(), c.winfo_height()
        if width < 50 or height < 50: # window not sized yet
            return
        pos = self.positions(width, height)

        for e in self.edges.itertuples(): # all roads in grey, guessed ones dashed
            guess = e.source == "guess"
            c.create_line(*pos[e.site_a], *pos[e.site_b], fill=GUESS_GREY if guess else ROAD_GREY, width=2, dash=(4, 4) if guess else ())

        chosen = self.selected()
        order = [i for i in range(len(self.routes)) if i != chosen] + ([chosen] if chosen is not None else [])
        for i in order: # selected route last so it sits on top
            points = [coord for site in self.routes[i][1] for coord in pos[site]]
            c.create_line(*points, fill=ROUTE_COLOURS[i], width=8 if i == chosen else 3, capstyle=tk.ROUND, joinstyle=tk.ROUND)

        origin, dest = self.site_id(self.origin), self.site_id(self.dest)
        for site, (x, y) in pos.items():
            outline, line_width = (ORIGIN_OUTLINE, 4) if site == origin else (DEST_OUTLINE, 4) if site == dest else (NODE_OUTLINE, 1.5)
            c.create_oval(x - NODE_R, y - NODE_R, x + NODE_R, y + NODE_R, fill=NODE_FILL, outline=outline, width=line_width)
            c.create_text(x + NODE_R + 3, y - NODE_R, text=str(site), anchor="sw", font=("Arial", 8))

        c.create_text(15, height - 12, anchor="w", font=("Arial", 9),
                      text="Blue ring = origin  |  Red ring = destination  |  Dashed = guessed road")


def main():
    os.chdir(HERE) # the backend uses relative paths, same as main.py
    root = tk.Tk()
    App(root, load_config())
    root.mainloop()


if __name__ == "__main__":
    main()