"""A 2-D picture of the paper embeddings, coloured by S3 community.

The S3 map is built on the cosine similarity of the title + abstract embeddings, and the
silhouette score measures how well the communities separate in that space. This module draws
the same space in two dimensions so readers can see the separation the silhouette reports, the
papers outside the main communities, and papers that sit between communities.

    UMAP  (umap-learn installed: pip install "s3-bibliometric[umap]"): cosine metric, 15 neighbours,
          min_dist 0.1, seeded with S3Config.seed. Keeps local neighbourhoods; distances between
          far-apart groups are not meaningful.
    PCA   (otherwise, or method="pca"): the two directions of largest variance. Linear, so groups
          overlap more than in UMAP.

Either way the picture is an illustration: the numbers (silhouette, neighbour agreement) are
computed on the full embeddings, not on the 2-D positions.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

N_NEIGHBORS = 15
MIN_DIST = 0.1


def _unit(emb: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.where(n == 0, 1.0, n)


def project(emb: np.ndarray, method: str = "auto", seed: int = 42) -> tuple[np.ndarray, str]:
    """2-D coordinates of the embeddings and the method used ("UMAP" or "PCA")."""
    method = method.lower()
    if method not in ("auto", "umap", "pca"):
        raise ValueError(f"method must be 'auto', 'umap' or 'pca', not {method!r}")
    n = len(emb)
    if method in ("auto", "umap") and n > N_NEIGHBORS + 1:
        try:
            from umap import UMAP
        except ImportError:
            if method == "umap":
                raise ImportError('UMAP needs umap-learn: pip install "s3-bibliometric[umap]"') from None
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xy = UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, metric="cosine",
                          random_state=seed).fit_transform(emb)
            return np.asarray(xy, dtype=float), "UMAP"
    from sklearn.decomposition import PCA
    k = min(2, n, emb.shape[1])
    xy = PCA(n_components=k, random_state=seed).fit_transform(_unit(emb))
    if k < 2:
        xy = np.hstack([xy, np.zeros((n, 2 - k))])
    return np.asarray(xy, dtype=float), "PCA"


def embedding_map(res, method: str = "auto") -> pd.DataFrame:
    """One row per paper: x, y, community (NaN outside the main communities), title, year.

    The method used is in `df.attrs["method"]`. Cached on the result per method."""
    cache = res.__dict__.setdefault("_embedding_map", {})
    if method in cache:
        return cache[method]
    xy, used = project(res.embeddings, method, res.config.seed)
    P = res.papers
    main = set(int(c) for c in res.communities()["community"])
    comm = pd.to_numeric(P["community"], errors="coerce")
    comm = comm.where(comm.isin(main))
    df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "community": comm.to_numpy(),
                       "title": P["title"].astype(str).to_numpy() if "title" in P else "",
                       "year": pd.to_numeric(P.get("year"), errors="coerce").to_numpy()
                       if "year" in P else np.nan},
                      index=P.index)
    df.attrs["method"] = used
    cache[method] = cache[used.lower()] = df
    return df


def separation(res) -> pd.DataFrame:
    """Per main community: papers, mean silhouette and neighbour agreement on the embeddings.

    silhouette            mean silhouette (cosine) of the community's papers, against the other
                          main communities
    neighbour agreement   share of each paper's k nearest papers (cosine, among all papers in
                          main communities) that are in its own community, k = 15 or the
                          community size - 1 if smaller
    agreement by chance   the same share if the community's papers were spread at random
    """
    from sklearn.metrics import silhouette_samples
    from sklearn.neighbors import NearestNeighbors
    comm = pd.to_numeric(res.papers["community"], errors="coerce")
    ids = [int(c) for c in res.communities()["community"]]
    m = comm.isin(ids).to_numpy()
    lab = comm.to_numpy()[m].astype(int)
    emb = _unit(res.embeddings[m])
    if len(set(lab)) < 2 or len(lab) <= len(set(lab)):
        return pd.DataFrame()
    sil = silhouette_samples(emb, lab, metric="cosine")
    k = min(N_NEIGHBORS, len(lab) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(emb)
    _, idx = nn.kneighbors(emb)
    size = pd.Series(lab).map(pd.Series(lab).value_counts()).to_numpy()
    kk = np.minimum(k, size - 1)  # a community of 3 can have at most 2 neighbours of its own
    same = lab[idx[:, 1:]] == lab[:, None]
    agree = np.array([same[i, :kk[i]].mean() if kk[i] else np.nan for i in range(len(lab))])
    chance = (size - 1) / (len(lab) - 1)
    rows = []
    for c in ids:
        s = lab == c
        if s.any():
            rows.append({"community": c, "papers": int(s.sum()),
                         "silhouette": round(float(sil[s].mean()), 3),
                         "neighbour agreement": round(float(np.nanmean(agree[s])), 3) if kk[s].any() else np.nan,
                         "agreement by chance": round(float(chance[s][0]), 3)})
    return pd.DataFrame(rows)


def chart(res, method: str = "auto", names: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
    """The report's scatter chart spec for the embedding map."""
    df = embedding_map(res, method)
    used = df.attrs["method"]
    counts = df["community"].value_counts()
    legend = [{"id": int(c), "name": (names or {}).get(int(c), f"C{int(c)}"), "n": int(counts.get(c, 0))}
              for c in res.communities()["community"]]
    pts: List[list] = []
    for x, y, c, t, yr in df[["x", "y", "community", "title", "year"]].itertuples(index=False):
        pts.append([round(float(x), 4), round(float(y), 4), -1 if pd.isna(c) else int(c),
                    str(t)[:90], None if pd.isna(yr) else int(yr)])
    return {"type": "scatter", "points": pts, "legend": legend, "method": used,
            "title": f"The papers' embeddings in two dimensions ({used}), coloured by S3 community"}
