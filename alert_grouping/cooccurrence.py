"""Historical time co-occurrence of alert types (patent module two).

Implements the co-occurrence leg of the patent's alert-merging module:

  T_ij = max{ N(c_i, c_j) / N(c_i),  N(c_i, c_j) / N(c_j) }

where N(c) is the number of historical sliding windows that contain alert
type c and N(c_i, c_j) is the number of windows containing both types.
"""

from collections import Counter
from itertools import combinations
from bisect import bisect_left

import numpy as np


class TimeCooccurrenceModel:
    """Window-wise historical co-occurrence statistics of alert types.

    Args:
        window_seconds (float): Length of the historical time window.
        step_seconds (float): Sliding step between consecutive windows.
    """

    def __init__(self, window_seconds=600.0, step_seconds=60.0):
        if not np.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be finite and positive.")
        if not np.isfinite(step_seconds) or step_seconds <= 0:
            raise ValueError("step_seconds must be finite and positive.")
        self.window_seconds = float(window_seconds)
        self.step_seconds = float(step_seconds)
        self.type_window_counts = Counter()
        self.pair_window_counts = Counter()
        self.n_windows = 0

    def fit(self, alerts, type_field="short", time_field="raw_time"):
        """Count type occurrences over historical sliding windows.

        Args:
            alerts (list[dict]): Historical alerts; must contain the type and
                time fields. Timestamps must be finite and share a timeline.
            type_field (str): Field used as the alert type (c_i).
            time_field (str): Numeric timestamp field.

        Returns:
            TimeCooccurrenceModel: self, for chaining.
        """
        if not isinstance(alerts, list) or any(
            not isinstance(row, dict) for row in alerts
        ):
            raise ValueError("Expected a JSON array of alert objects.")
        records = []
        for row in alerts:
            value = row.get(type_field)
            timestamp = row.get(time_field)
            if value is None or not np.isfinite(float(timestamp)):
                raise ValueError("Alerts need a type field and finite timestamps.")
            records.append((float(timestamp), str(value)))
        if not records:
            return self
        records.sort()
        times = [t for t, _ in records]
        types = [t for _, t in records]
        start = times[0]
        end = times[-1]
        n_buckets = max(1, int(np.floor((end - start) / self.step_seconds)) + 1)
        bucket_starts = start + np.arange(n_buckets) * self.step_seconds
        for bucket in range(n_buckets):
            window_start = bucket_starts[bucket]
            window_end = window_start + self.window_seconds
            left = bisect_left(times, window_start)
            right = bisect_left(times, window_end)
            present = set(types[left:right])
            self.n_windows += 1
            for value in present:
                self.type_window_counts[value] += 1
            for value_a, value_b in combinations(sorted(present), 2):
                self.pair_window_counts[(value_a, value_b)] += 1
        return self

    def cooccurrence(self, type_a, type_b):
        """T_ij = max of the two directional conditional probabilities.

        Args:
            type_a (str): Alert type of alert a_i.
            type_b (str): Alert type of alert a_j.

        Returns:
            float: Co-occurrence degree in [0, 1]; 1.0 for identical types,
                0.0 when either type has never been observed.
        """
        type_a, type_b = str(type_a), str(type_b)
        if type_a == type_b:
            return 1.0
        if type_a > type_b:
            type_a, type_b = type_b, type_a
        count_a = self.type_window_counts.get(type_a, 0)
        count_b = self.type_window_counts.get(type_b, 0)
        count_pair = self.pair_window_counts.get((type_a, type_b), 0)
        if count_a == 0 or count_b == 0:
            return 0.0
        return max(count_pair / count_a, count_pair / count_b)

    def pairwise_matrix(self, types):
        """Pairwise co-occurrence matrix for a sequence of alert types.

        Args:
            types (Sequence[str]): One alert type per alert.

        Returns:
            np.ndarray: Symmetric (n, n) matrix; diagonal is 1.0.
        """
        types = [str(value) for value in types]
        n = len(types)
        matrix = np.ones((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                value = self.cooccurrence(types[i], types[j])
                matrix[i, j] = value
                matrix[j, i] = value
        return matrix
