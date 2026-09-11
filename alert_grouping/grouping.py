"""Patent module two (告警归并): semantic distance and time co-occurrence grouping.

Implements the patent's alert-merging module:

  D_sem(a_i, a_j) = 1 - cos(e_i, e_j)

where e_i is the contextual semantic embedding of alert a_i produced by the
alert semantic encoding model (:mod:`alert_grouping.semantic`), and the fused
association score

  R(a_i, a_j) = T_ij - alpha * (D_sem - D_min) / (D_max - D_min)

where T_ij is the historical time co-occurrence degree from
:mod:`alert_grouping.cooccurrence`. Alerts whose score passes the association
threshold are connected by an edge and the connected components of the
resulting graph form alert events.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def semantic_distance(vector_a, vector_b):
    """Cosine distance between two semantic vectors.

    Args:
        vector_a (np.ndarray): Semantic embedding of alert a_i.
        vector_b (np.ndarray): Semantic embedding of alert a_j.

    Returns:
        float: D_sem(a_i, a_j) in [0, 2]; 0 for parallel vectors.
    """
    vector_a = np.asarray(vector_a, dtype=np.float64)
    vector_b = np.asarray(vector_b, dtype=np.float64)
    if vector_a.ndim != 1 or vector_b.ndim != 1 or vector_a.size != vector_b.size:
        raise ValueError("Both vectors must be 1-D and equally sized.")
    norm_a, norm_b = np.linalg.norm(vector_a), np.linalg.norm(vector_b)
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Semantic vectors must not be zero vectors.")
    cosine = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
    return 1.0 - min(max(cosine, -1.0), 1.0)


def semantic_distance_matrix(vectors):
    """Pairwise cosine-distance matrix for a set of semantic vectors.

    Args:
        vectors (np.ndarray): Array of shape (n_alerts, dim).

    Returns:
        np.ndarray: Symmetric (n_alerts, n_alerts) matrix; diagonal is 0.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError("Expected a 2-D array of semantic vectors.")
    if len(vectors) == 0:
        return np.empty((0, 0), dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not (norms > 0).all():
        raise ValueError("Semantic vectors must not be zero vectors.")
    normed = vectors / norms
    cosine = normed @ normed.T
    np.fill_diagonal(cosine, 1.0)
    return 1.0 - np.clip(cosine, -1.0, 1.0)


def group_by_semantic_distance(distances, threshold):
    """Group alerts by thresholding semantic distances.

    Alerts with D_sem <= threshold are linked by an edge; each connected
    component of the resulting graph is one alert event (patent step:
    关联边 + 连通分量). Singleton components are returned as their own groups
    so that every input alert belongs to exactly one group.

    Args:
        distances (np.ndarray): Pairwise distance matrix of shape (n, n).
        threshold (float): Association threshold theta in [0, 2].

    Returns:
        tuple[np.ndarray, np.ndarray]:
            group labels of shape (n,), one integer per alert, and
            the boolean adjacency matrix of shape (n, n).
    """
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be a square matrix.")
    if not 0.0 <= threshold <= 2.0:
        raise ValueError("threshold must be in [0, 2].")
    adjacency = distances <= threshold
    return _connected_components(adjacency)


def combined_association_scores(distances, cooccurrence_matrix, alpha=0.5):
    """Fuse semantic distance and time co-occurrence (patent formula).

    R(a_i, a_j) = T_ij - alpha * (D_sem - D_min) / (D_max - D_min)

    where D_min/D_max are taken over all currently compared alert pairs and
    T_ij is the historical time co-occurrence degree. Higher R means a
    stronger association.

    Args:
        distances (np.ndarray): Pairwise semantic distance matrix (n, n).
        cooccurrence_matrix (np.ndarray): Pairwise T_ij matrix (n, n).
        alpha (float): Weight of the normalized semantic distance in [0, 1].

    Returns:
        np.ndarray: Combined association score matrix of shape (n, n).
    """
    distances = np.asarray(distances, dtype=np.float64)
    cooccurrence_matrix = np.asarray(cooccurrence_matrix, dtype=np.float64)
    if distances.shape != cooccurrence_matrix.shape or distances.ndim != 2:
        raise ValueError("distances and co-occurrence matrix must be equally sized squares.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    if distances.size == 0:
        return distances.copy()
    upper = distances[np.triu_indices_from(distances, 1)]
    d_min = float(upper.min()) if upper.size else 0.0
    d_max = float(upper.max()) if upper.size else 0.0
    span = d_max - d_min
    normalized = (distances - d_min) / span if span > 0 else np.zeros_like(distances)
    return cooccurrence_matrix - alpha * normalized


def group_by_association_score(scores, threshold):
    """Group alerts by thresholding combined association scores.

    Alerts with R >= threshold are linked by an edge and each connected
    component is one alert event.

    Args:
        scores (np.ndarray): Combined association score matrix (n, n).
        threshold (float): Association threshold theta.

    Returns:
        tuple[np.ndarray, np.ndarray]: Group labels and adjacency matrix.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be a square matrix.")
    adjacency = scores >= threshold
    return _connected_components(adjacency)


def _connected_components(adjacency):
    """Drop self loops and return connected-component labels and adjacency."""
    np.fill_diagonal(adjacency, False)
    n_components, labels = connected_components(
        csr_matrix(adjacency), directed=False
    )
    return labels, adjacency


def group_high_priority_alerts(alert_rows, encoder, threshold, cooccurrence=None,
                               alpha=0.5, type_field="short"):
    """Encode high-priority alerts and merge them into alert events.

    Semantic leg of the patent's alert-merging module; when a
    :class:`TimeCooccurrenceModel` is supplied, the semantic distance and the
    historical time co-occurrence are fused into the combined association
    score R = T - alpha * normalized(D_sem), which is then thresholded.

    Args:
        alert_rows (list[dict]): High-priority alerts; each row needs the
            fields consumed by the encoder ("short", "host", "raw_time").
        encoder (AlertSemanticEncoder): Semantic encoder producing one
            contextual vector per alert in input order.
        threshold (float): Association threshold theta; an upper bound on the
            semantic distance, or a lower bound on the combined score R.
        cooccurrence (TimeCooccurrenceModel | None): Optional fitted historical
            co-occurrence model enabling the fused score.
        alpha (float): Weight of the normalized semantic distance in [0, 1].
        type_field (str): Alert type field used for co-occurrence lookup.

    Returns:
        tuple[list[list[int]], np.ndarray, np.ndarray]:
            groups as lists of input indices, the group label per alert, and
            the pairwise semantic distance matrix.
    """
    if not isinstance(alert_rows, list):
        raise ValueError("alert_rows must be a list of alert dictionaries.")
    vectors = encoder.encode(alert_rows)
    distances = semantic_distance_matrix(vectors)
    if cooccurrence is None:
        labels, _ = group_by_semantic_distance(distances, threshold)
    else:
        types = [row[type_field] for row in alert_rows]
        cooccurrence_matrix = cooccurrence.pairwise_matrix(types)
        scores = combined_association_scores(distances, cooccurrence_matrix, alpha)
        labels, _ = group_by_association_score(scores, threshold)
    groups = [[] for _ in range(labels.max() + 1 if len(labels) else 0)]
    for index, label in enumerate(labels):
        groups[int(label)].append(index)
    return groups, labels, distances
