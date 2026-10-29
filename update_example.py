"""Embed a saved CBCR solution and its cached street paths, preserving other cities."""

import argparse
import hashlib
import io
import json
import pickle
import re
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import networkx as nx
import osmnx as ox


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=100)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--instance", type=Path)
    args = parser.parse_args()
    if args.nodes < 3:
        parser.error("At least three nodes are required")
    seed = args.seed if args.seed is not None else 2 * (args.nodes - 10) + 1
    root = Path(__file__).resolve().parent
    code = root.parent / "cvmrptw_code"
    sys.path.insert(0, str(code.parent))
    from cvmrptw_code import instance as loader
    from cvmrptw_code.metrics import quality_metrics
    from cvmrptw_code.sampling import sampling_signature

    result_path = args.result or code / f"results/cbcr/n{args.nodes}_seed{seed}.json"
    record = json.loads(result_path.read_text())
    result = record["result"]
    if (
        record["summary"]["Algorithm"] != "cbcr"
        or record["instance"]["Nodes"] != args.nodes
        or not result.get("Solution Validity")
    ):
        raise ValueError(f"A validated CBCR result with n={args.nodes} is required")
    settings = {key: value for key, value in record["instance"].items() if key not in ("Nodes", "Customers")}
    key = hashlib.sha256(
        (json.dumps(settings, sort_keys=True) + sampling_signature()).encode()
    ).hexdigest()
    cached = args.instance or code / "results/heuristic_cases" / f"{key}.pkl"
    if cached.exists():
        instance = pickle.loads(cached.read_bytes())
    else:
        with tempfile.TemporaryDirectory(prefix="cbcr-example-") as directory:
            loader.PATH_CACHE_DIR = Path(directory)
            coordinate_cache = code / "tests/path_cache/coord_pairs.pkl"
            if coordinate_cache.exists():
                shutil.copyfile(coordinate_cache, loader.PATH_CACHE_DIR / "coord_pairs.pkl")
            with redirect_stdout(io.StringIO()):
                instance = loader.load_real_instance(**settings)
    if hashlib.sha256(pickle.dumps(instance)).hexdigest() != record["instance_sha256"]:
        raise ValueError("Rebuilt instance differs from the saved experiment")

    van_route = result["Van Route"]
    moped_routes = result["Moped Routes"]
    van_times = {int(node): time for node, time in result["Van Times"].items()}
    moped_times = [{int(node): time for node, time in times.items()} for times in result["Moped Times"]]
    metrics = quality_metrics(
        van_route, moped_routes, van_times, moped_times,
        instance["d"], instance["L_m"], instance["c_vn"], instance["c_m"],
    )
    clusters = {node: index for index, group in enumerate(record["details"]["clusters"]) for node in group}
    routes = []
    for vehicle, paths, arrivals in (
        ("van", [van_route], [van_times]),
        ("moped", moped_routes, moped_times),
    ):
        graph = ox.load_graphml(code / f"graphs/{settings['location']}_drive_{vehicle}.graphml")
        if vehicle == "van":
            van_graph = graph
        coordinates = [instance["coords"][node] for node in instance["V"]]
        snapped = ox.distance.nearest_nodes(graph, [point[0] for point in coordinates], [point[1] for point in coordinates])
        for index, (route, times) in enumerate(zip(paths, arrivals, strict=True)):
            segments = []
            for source, target in zip(route, route[1:]):
                path = nx.shortest_path(graph, snapped[source], snapped[target], weight="travel_time")
                points = []
                for u, v in zip(path, path[1:]):
                    edge = min(graph[u][v].values(), key=lambda data: data["travel_time"])
                    geometry = list(edge["geometry"].coords) if "geometry" in edge else [
                        (graph.nodes[u]["x"], graph.nodes[u]["y"]),
                        (graph.nodes[v]["x"], graph.nodes[v]["y"]),
                    ]
                    start = (graph.nodes[u]["x"], graph.nodes[u]["y"])
                    distance = lambda point: sum((a - b) ** 2 for a, b in zip(point, start))
                    if distance(geometry[-1]) < distance(geometry[0]):
                        geometry.reverse()
                    points.extend(geometry if not points else geometry[1:])
                if not points:
                    points = [instance["coords"][source], instance["coords"][target]]
                segments.append({
                    "source": source, "target": target,
                    "points": [[round(x, 7), round(y, 7)] for x, y in points],
                    "travel": instance["c_vn" if vehicle == "van" else "c_m"][source, target],
                })
            routes.append({"vehicle": vehicle, "index": index, "nodes": route, "arrivals": times, "segments": segments})

    page = root / "index.html"
    html = page.read_text()
    pattern = r'(<script type="application/json" id="solution-data">)(.*?)(</script>)'
    match = re.search(pattern, html, re.S)
    existing = json.loads(match.group(2))
    roads = next((case["roads"] for case in existing if case["city"] == settings["location"]), None)
    if roads is None:
        extent = list(instance["coords"].values()) + instance["sampling"]["boundary"]
        left, right = min(point[0] for point in extent), max(point[0] for point in extent)
        bottom, top = min(point[1] for point in extent), max(point[1] for point in extent)
        roads = []
        for source, target, edge in van_graph.edges(data=True):
            points = list(edge["geometry"].coords) if "geometry" in edge else [
                (van_graph.nodes[source]["x"], van_graph.nodes[source]["y"]),
                (van_graph.nodes[target]["x"], van_graph.nodes[target]["y"]),
            ]
            if any(left <= x <= right and bottom <= y <= top for x, y in points):
                roads.append([[round(x, 7), round(y, 7)] for x, y in points])
    data = {
        "city": settings["location"], "settings": settings,
        "seconds": result["Algorithm (s)"], "sampling": instance["sampling"],
        "metrics": metrics, "tct": result["f2"], "routes": routes,
        "nodes": [{
            "id": node, "xy": instance["coords"][node], "demand": instance["d"][node],
            "cluster": clusters.get(node), "a": instance["a"][node], "b": instance["b"][node],
            "service": instance["s_vn" if node in van_route else "s_m"][node],
            "van": node in van_route, "depot": node in (instance["V"][0], instance["V"][-1]),
        } for node in instance["V"]],
        "roads": roads,
    }
    cases = [case for case in existing if case["city"] != data["city"]] + [data]
    cases.sort(key=lambda case: case["city"])
    html = html[:match.start(2)] + json.dumps(cases, separators=(",", ":"), allow_nan=False) + html[match.end(2):]
    for identifier, count in (("node-count", len(cases[0]["nodes"])), ("customer-count", len(cases[0]["nodes"]) - 2)):
        html = re.sub(rf'(<span id="{identifier}">)\d+(</span>)', lambda match: f"{match[1]}{count}{match[2]}", html)
    page.write_text(html)
    print(f"Embedded {settings['location']} CBCR n={args.nodes}, seed={settings['seed']}, {len(moped_routes)} mopeds.")


if __name__ == "__main__":
    main()
