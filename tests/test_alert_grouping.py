import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from alert_grouping.data import FEATURES, sequence_slices
from alert_grouping.models import MaskedLangModelParams, MaskedLanguageModel
from alert_grouping.preprocessing import Vocabulary
from alert_grouping.train_mlm import masked_batch, forward_loss
from alert_grouping.semantic import AlertSemanticEncoder
from alert_grouping.cooccurrence import TimeCooccurrenceModel
from alert_grouping.grouping import group_high_priority_alerts


class SemanticTests(unittest.TestCase):
    def test_window_boundaries(self):
        scene = {"raw_time": np.array([0., 1., 599., 600., 601.])}
        chunks = list(sequence_slices(scene, context_size=2))
        self.assertEqual([(s.start, s.stop) for s in chunks],
                         [(0, 2), (2, 3), (3, 5)])

    def test_training_reload_and_label_independence(self):
        torch.set_num_threads(1)
        vocabs = {f: Vocabulary(min_freq=1) for f in FEATURES}
        for vocab in vocabs.values():
            vocab.build_from_iterator(["a", "b"])
        params = MaskedLangModelParams(id_suffix="test", n_heads=2, dim_per_head=16,
                                      features=FEATURES, targets=FEATURES, min_freq=1)
        model = MaskedLanguageModel(params, vocabs)
        sequence = {"short": np.array(["a"]), "host": np.array(["b"]),
                    "raw_time": np.array([1640000000.])}
        batch, indices = masked_batch(sequence, vocabs, 42)
        self.assertEqual(indices.shape, (2, 1))
        loss, _ = forward_loss(model, batch, indices, vocabs)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "report.json").write_text(json.dumps({"params": params.dict}))
            for f, vocab in vocabs.items():
                vocab.save(str(path / f"vocab_{f}.json"))
            torch.save(model.state_dict(), path / "model.pt")
            encoder = AlertSemanticEncoder(path)
            rows = [{"short": "a", "host": "b", "time": 1640000002.},
                    {"short": "unknown", "host": "a", "time": 1640000001.}]
            vectors = encoder.encode(rows)
            self.assertEqual(vectors.shape, (2, 32))
            self.assertTrue(np.isfinite(vectors).all())
            np.testing.assert_allclose(encoder.encode(rows[::-1]), vectors[::-1])
            np.testing.assert_array_equal(encoder.encode([
                dict(row, event_label="attack", time_label="stage") for row in rows
            ]), vectors)
            self.assertEqual(encoder.encode([]).shape, (0, 32))
            groups, labels, _ = group_high_priority_alerts(rows, encoder, 2.)
            self.assertEqual(groups, [[0, 1]])
            self.assertEqual(labels.tolist(), [0, 0])

    def test_cooccurrence_counts_and_fusion(self):
        model = TimeCooccurrenceModel(window_seconds=10, step_seconds=10).fit([
            {"short": "a", "raw_time": 0},
            {"short": "b", "raw_time": 1},
            {"short": "a", "raw_time": 10},
            {"short": "b", "raw_time": 20},
        ])
        self.assertEqual(model.n_windows, 3)
        self.assertEqual(model.cooccurrence("a", "b"), .5)
        self.assertEqual(model.cooccurrence("a", "unseen"), 0.)
        class Encoder:
            def encode(self, rows):
                return np.array([[1., 0.], [0., 1.]])
        groups, _, _ = group_high_priority_alerts(
            [{"short": "a"}, {"short": "b"}], Encoder(), .5,
            cooccurrence=model,
        )
        self.assertEqual(groups, [[0, 1]])


if __name__ == "__main__":
    unittest.main()
