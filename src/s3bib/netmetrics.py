"""Network indicators: the S3 network as a whole, its communities, and each paper's position.

Network level (network_level), for the map network at the threshold chosen in Q1 and, at level 2,
the coupling, co-citation, direct-citation and co-authorship layers of s3bib.layers:
    papers, links, density, average degree, papers with links, connected groups, share of papers
    in the largest group, clustering (transitivity), average local clustering, average path length
    and diameter of the largest group, degree centralization (Freeman 1978), degree assortativity
    (Newman 2002), modularity of the S3 communities, and a small-world check against a random
    network with the same papers and links (Watts & Strogatz 1998; Humphries & Gurney 2008).
Communities as subnetworks (community_level): links inside and to other communities, density
    inside, average degree inside, clustering, hubs and connectors.
Node level (node_level): degree, strength (sum of link similarities), eigenvector centrality (as in
    Q7), betweenness, harmonic closeness, PageRank, local clustering, k-core, participation
    coefficient P and within-community degree z, and the role these two give (Guimerà & Amaral 2005).

All measures are exact (igraph), except the average path length and diameter of a largest group
of more than 5,000 papers, which are estimated from 1,000 random start papers.

Reading: the S3 network is a similarity network cut at a threshold. A paper's degree depends on
that threshold, betweenness means "sits between topics", not "passes information along", and
clustering is high by construction (two papers similar to a third tend to be similar to each
other). Flow readings belong to the citation network (Q10, Q37).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

EXACT_PATHS = 5000     # largest group up to this size: exact path lengths
PATH_SAMPLE = 1000     # otherwise: start papers sampled for path lengths
HUB_Z = 2.5            # Guimerà & Amaral (2005): hubs have within-community degree z >= 2.5
EULER = 0.5772156649

ROLES = ("ultra-peripheral", "peripheral", "connector", "kinless",
         "provincial hub", "connector hub", "kinless hub")


# ------------------------------------------------------------------ graphs --
def _map_graph(res):
    """The map network (res.graph()) as an igraph Graph with the link similarities as weights."""
    import igraph as ig
    G = res.graph()
    edges = [(int(u), int(v)) for u, v in G.edges()]
    g = ig.Graph(n=len(res.papers), edges=edges)
    g.es["weight"] = [float(d.get("weight", 1.0)) for _, _, d in G.edges(data=True)]
    return g


def _layer_graph(keys: np.ndarray, n: int):
    import igraph as ig
    keys = np.asarray(keys, dtype=np.int64)
    return ig.Graph(n=n, edges=np.column_stack([keys // n, keys % n]).tolist())


def _membership(res) -> np.ndarray:
    """S3 community per paper; papers outside the main communities get one group each."""
    comm = pd.to_numeric(res.papers["community"], errors="coerce")
    main = set(int(c) for c in res.communities()["community"])
    comm = comm.where(comm.isin(main))
    mem = np.empty(len(comm), dtype=int)
    nxt = int(comm.max()) + 1 if comm.notna().any() else 0
    for i, c in enumerate(comm.to_numpy()):
        if pd.isna(c):
            mem[i], nxt = nxt, nxt + 1
        else:
            mem[i] = int(c)
    return mem


def map_label(res) -> str:
    return "hybrid map" if res.config.map_weights else "S3 map"


# ----------------------------------------------------------- network level --
def _paths(g) -> tuple:
    """(average path length, diameter, estimated?) of a connected graph."""
    n = g.vcount()
    if n < 2:
        return np.nan, np.nan, False
    if n <= EXACT_PATHS:
        return float(g.average_path_length(directed=False)), float(g.diameter(directed=False)), False
    rng = np.random.default_rng(42)
    src = rng.choice(n, size=PATH_SAMPLE, replace=False).tolist()
    d = np.asarray(g.distances(source=src), dtype=float)
    d = d[np.isfinite(d) & (d > 0)]
    return float(d.mean()), float(d.max()), True


def _net_stats(g, mem: np.ndarray, weights: Optional[str] = None) -> Dict[str, float]:
    n, m = g.vcount(), g.ecount()
    deg = np.asarray(g.degree(), dtype=float)
    linked = deg > 0
    comps = g.connected_components()
    sizes = np.asarray(comps.sizes())
    out: Dict[str, float] = {
        "papers": n, "links": m,
        "density": 2 * m / (n * (n - 1)) if n > 1 else np.nan,
        "average degree": 2 * m / n if n else np.nan,
        "papers with links": int(linked.sum()),
        "connected groups (papers with links)": int((sizes > 1).sum()),
        "share of papers in the largest group": float(sizes.max() / n) if n and m else 0.0,
    }
    out["clustering (transitivity)"] = float(g.transitivity_undirected()) if m else np.nan
    loc = np.asarray(g.transitivity_local_undirected(mode="zero"), dtype=float)
    out["average local clustering (papers with links)"] = float(loc[linked].mean()) if linked.any() else np.nan
    lcc = comps.giant() if m else None
    L, D, est = _paths(lcc) if lcc is not None else (np.nan, np.nan, False)
    out["average path length (largest group)"] = L
    out["diameter (largest group)"] = D
    out["path lengths estimated from a sample"] = est
    out["degree centralization"] = float((deg.max() - deg).sum() / ((n - 1) * (n - 2))) if n > 2 and m else np.nan
    try:
        r = g.assortativity_degree(directed=False)
        out["degree assortativity"] = float(r) if r is not None and math.isfinite(r) else np.nan
    except Exception:
        out["degree assortativity"] = np.nan
    try:
        out["modularity of the S3 communities"] = float(g.modularity(mem.tolist(), weights=weights)) if m else np.nan
    except Exception:
        out["modularity of the S3 communities"] = np.nan
    # small-world check on the largest group: a random network with the same papers and links
    # has clustering k/(n-1) and average path length (ln n - γ)/ln k + 1/2 (Fronczak et al. 2004)
    cr = lr = sig = np.nan
    if lcc is not None and lcc.vcount() > 2:
        nl, ml = lcc.vcount(), lcc.ecount()
        k = 2 * ml / nl
        if k > 1:
            c_rand = k / (nl - 1)
            l_rand = (math.log(nl) - EULER) / math.log(k) + 0.5
            c_l = lcc.transitivity_undirected()
            if c_rand > 0 and math.isfinite(c_l):
                cr = c_l / c_rand
            if l_rand > 0 and math.isfinite(L):
                lr = L / l_rand
            if math.isfinite(cr) and math.isfinite(lr) and lr > 0:
                sig = cr / lr
    out["clustering / random network"] = cr
    out["path length / random network"] = lr
    out["small-world sigma"] = sig
    return out


def network_level(res, layers: Optional[bool] = None) -> pd.DataFrame:
    """Rows: indicators; columns: the map network and (layers=True, default at level 2) the
    coupling, co-citation, direct-citation and co-authorship layers. Cached."""
    if layers is None:
        layers = int(getattr(res.config, "level", 2)) >= 2
    cache = res.__dict__.setdefault("_netmetrics", {})
    key = ("network", bool(layers))
    if key in cache:
        return cache[key]
    mem = _membership(res)
    cols = {map_label(res): _net_stats(_map_graph(res), mem, weights="weight")}
    if layers:
        from .layers import layers as get_layers
        n = len(res.papers)
        for name, keys in get_layers(res).items():
            if name != "text" and len(keys):
                cols[name] = _net_stats(_layer_graph(keys, n), mem)
    df = pd.DataFrame(cols)
    cache[key] = df
    return df


# -------------------------------------------------------------- node level --
def _role(z: float, p: float) -> str:
    if z >= HUB_Z:
        return "provincial hub" if p <= 0.30 else "connector hub" if p <= 0.75 else "kinless hub"
    return ("ultra-peripheral" if p <= 0.05 else "peripheral" if p <= 0.62
            else "connector" if p <= 0.80 else "kinless")


def node_level(res) -> pd.DataFrame:
    """One row per paper (same index as res.papers): title, year, community, cited_by and the
    node indicators of the map network, with each paper's role. Cached."""
    cache = res.__dict__.setdefault("_netmetrics", {})
    if "nodes" in cache:
        return cache["nodes"]
    P = res.papers
    n = len(P)
    g = _map_graph(res)
    mem = _membership(res)
    main = set(int(c) for c in res.communities()["community"])
    in_main = np.array([m in main for m in mem])
    deg = np.asarray(g.degree(), dtype=float)
    strength = np.asarray(g.strength(weights="weight"), dtype=float) if g.ecount() else np.zeros(n)
    bet = np.asarray(g.betweenness(directed=False), dtype=float)
    if n > 2:
        bet = bet / ((n - 1) * (n - 2) / 2)  # normalised as in networkx (and Q7)
    harm = np.asarray(g.harmonic_centrality(normalized=True), dtype=float)
    pr = np.asarray(g.pagerank(weights="weight" if g.ecount() else None), dtype=float)
    loc = np.asarray(g.transitivity_local_undirected(mode="zero"), dtype=float)
    core = np.asarray(g.coreness(), dtype=int)

    # participation P and within-community degree z (Guimerà & Amaral 2005)
    e = np.asarray(g.get_edgelist(), dtype=int).reshape(-1, 2)
    A = sp.csr_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))
    M = sp.csr_matrix((np.ones(n), (np.arange(n), mem)), shape=(n, int(mem.max()) + 1))
    K = (A @ M).tocsr()                                # links of each paper to each community
    sq = np.asarray(K.multiply(K).sum(axis=1)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        part = np.where(deg > 0, 1 - sq / deg ** 2, 0.0)
    k_in = np.asarray(K[np.arange(n), mem]).ravel()
    z = np.full(n, np.nan)
    for c in main:
        s = mem == c
        sd = k_in[s].std()
        z[s] = (k_in[s] - k_in[s].mean()) / sd if sd > 0 else 0.0
    role = [("outlier (no links)" if deg[i] == 0 else
             "outside the main communities" if not in_main[i] else _role(z[i], part[i])) for i in range(n)]

    comm = pd.to_numeric(P["community"], errors="coerce").where(pd.Series(in_main, index=P.index))
    df = pd.DataFrame({
        "title": P["title"].astype(str) if "title" in P else "",
        "year": pd.to_numeric(P["year"], errors="coerce") if "year" in P else np.nan,
        "community": comm.astype("Int64"),
        "cited_by": P["cited_by"] if "cited_by" in P else np.nan,
        "degree": deg.astype(int), "strength": strength,
        "eigenvector": P["eigen_centrality"].astype(float) if "eigen_centrality" in P else np.nan,
        "betweenness": bet, "harmonic closeness": harm, "pagerank": pr,
        "local clustering": loc, "k-core": core,
        "participation": part, "within-community z": z, "role": role}, index=P.index)
    for c in ("eid", "doi"):
        if c in P:
            df.insert(0, c, P[c].to_numpy())
    cache["nodes"] = df
    return df


def community_level(res) -> pd.DataFrame:
    """One row per main community: the community as a subnetwork of the map network. Cached."""
    cache = res.__dict__.setdefault("_netmetrics", {})
    if "communities" in cache:
        return cache["communities"]
    nodes = node_level(res)
    g = _map_graph(res)
    mem = _membership(res)
    e = np.asarray(g.get_edgelist(), dtype=int).reshape(-1, 2)
    a, b = (mem[e[:, 0]], mem[e[:, 1]]) if len(e) else (np.zeros(0, int), np.zeros(0, int))
    rows = []
    for c in res.communities()["community"]:
        c = int(c)
        s = int((mem == c).sum())
        inside = int(((a == c) & (b == c)).sum())
        out = int(((a == c) ^ (b == c)).sum())
        sub = nodes[mem == c]
        rows.append({"community": c, "papers": s, "links inside": inside,
                     "density inside": 2 * inside / (s * (s - 1)) if s > 1 else np.nan,
                     "average degree inside": 2 * inside / s if s else np.nan,
                     "links to other communities": out,
                     "share of links outside": out / (inside + out) if inside + out else np.nan,
                     "average local clustering": float(sub["local clustering"].mean()),
                     "hubs": int(sub["role"].str.contains("hub").sum()),
                     "connectors": int(sub["role"].isin(["connector", "kinless", "connector hub",
                                                         "kinless hub"]).sum())})
    df = pd.DataFrame(rows)
    cache["communities"] = df
    return df


DEGREE_BINS = [(0, 0, "0 (outliers)"), (1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 9, "5-9"),
               (10, 19, "10-19"), (20, 49, "20-49"), (50, 99, "50-99"), (100, 10 ** 9, "100+")]


def degree_distribution(res) -> pd.DataFrame:
    deg = node_level(res)["degree"].to_numpy()
    rows = [{"links": lab, "papers": int(((deg >= lo) & (deg <= hi)).sum())} for lo, hi, lab in DEGREE_BINS]
    while len(rows) > 1 and rows[-1]["papers"] == 0:
        rows.pop()
    return pd.DataFrame(rows)
