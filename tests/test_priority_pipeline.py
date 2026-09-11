import copy
import importlib
import json
import unittest

import alert_priority.main as priority


def alert(timestamp, host="h", short="s", **extra):
    return dict(time=timestamp, raw_time=timestamp, ip="10.0.0.1",
                host=host, short=short, name=short, **extra)


class PriorityApiTests(unittest.TestCase):
    def rank(self, rows, **kwargs):
        self.assertTrue(callable(getattr(priority, "prioritize_alerts", None)),
                        "The JSON-array priority API is missing")
        return priority.prioritize_alerts(rows, **kwargs)

    def test_empty_and_single_unlabelled_alert(self):
        self.assertEqual(self.rank([])["ranked"], [])
        result = self.rank([alert(1)])
        self.assertEqual(len(result["ranked"]), 1)
        self.assertEqual(len(result["high_priority"]) + len(result["low_priority"]), 1)
        json.dumps(result, allow_nan=False)

    def test_merging_sorting_partition_and_original_members(self):
        rows = [alert(0, alert_id="a"), alert(10, alert_id="b")]
        rows += [alert(i * 20, host=f"h{i}", short=f"s{i}", alert_id=f"x{i}")
                 for i in range(1, 10)]
        before = copy.deepcopy(rows)
        result = self.rank(rows, contamination=0.2)
        self.assertEqual(rows, before)
        ranked = result["ranked"]
        scores = [item["anomaly_score"] for item in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(len(ranked), len(rows) - 1)
        self.assertCountEqual([i for item in ranked for i in item["member_ids"]],
                              [row["alert_id"] for row in rows])
        self.assertIn(["a", "b"], [item["member_ids"] for item in ranked])
        for key, value in (("high_priority", "high"), ("low_priority", "low")):
            self.assertEqual(result[key], [r for r in ranked if r["priority"] == value])
        json.dumps(result, allow_nan=False)

    def test_labels_do_not_affect_callable_results(self):
        rows = [alert(i, host=str(i), event_label="-") for i in range(6)]
        first = self.rank(rows)
        for row in rows:
            row["event_label"] = "attack"
        second = self.rank(rows)
        self.assertEqual([r["anomaly_score"] for r in first["ranked"]],
                         [r["anomaly_score"] for r in second["ranked"]])

    def test_invalid_json_and_times(self):
        for rows in ({}, [1], [alert(float("nan"))], [alert(float("inf"))]):
            with self.subTest(rows=rows), self.assertRaises((ValueError, TypeError)):
                self.rank(rows)

    def test_priority_count_summary_distinguishes_group_totals_and_correct_groups(self):
        summary = priority.priority_count_summary([8, 2, 1, 4])
        self.assertEqual(summary, {
            "total_groups": 15,
            "attack_groups": 5,
            "noise_groups": 10,
            "high_priority_groups": 6,
            "low_priority_groups": 9,
            "correctly_classified_groups": 12,
            "incorrectly_classified_groups": 3,
            "attack_groups_correct_high": 4,
            "noise_groups_correct_low": 8,
        })

    def test_fixed_contamination_is_resolved_without_labels(self):
        self.assertEqual(priority.resolve_contamination("0.03", [1, 0, 1]), 0.03)
        self.assertEqual(priority.resolve_contamination("auto", [1, 0, 1]), "auto")
        self.assertAlmostEqual(priority.resolve_contamination("label_rate", [1, 0, 1]), 2 / 3)
        with self.assertRaises(ValueError):
            priority.resolve_contamination("0.51", [1, 0])


class ScoreAggregationTests(unittest.TestCase):
    def accumulator(self):
        pipeline = importlib.import_module("main")
        self.assertTrue(hasattr(pipeline, "WindowScoreAccumulator"), "Score averaging is missing")
        return pipeline.WindowScoreAccumulator()

    def window(self, start, rows, groups, threshold=0.5):
        ranked = [dict(member_ids=ids, anomaly_score=score) for ids, score in groups]
        return dict(window_start=start, window_end=start + 600,
                    priority_result=dict(ranked=ranked, score_threshold=threshold))

    def test_average_each_original_member_once_per_present_window(self):
        acc = self.accumulator()
        a, b, c = [alert(100, host=x, alert_id=x) for x in "abc"]
        first = self.window(0, [a, b, c], [(["a", "b"], 0.8), (["c"], 0.2)])
        second = self.window(60, [b, c], [(["b", "c"], 0.4)], threshold=0.6)
        acc.add(first, [a, b, c])
        acc.add(second, [b, c])
        result = acc.finalize()
        self.assertEqual([r["alert_id"] for r in result["ranked"]], ["a", "b", "c"])
        by_id = {r["alert_id"]: r for r in result["ranked"]}
        for aid, score, count in (("a", 0.8, 1), ("b", 0.6, 2), ("c", 0.3, 2)):
            self.assertAlmostEqual(by_id[aid]["anomaly_score"], score)
            self.assertAlmostEqual(by_id[aid]["score_sum"], score * count)
            self.assertEqual(by_id[aid]["window_count"], count)
            self.assertEqual(by_id[aid]["alert"]["host"], aid)
        self.assertAlmostEqual(by_id["b"]["score_threshold"], 0.55)
        self.assertEqual([r["alert_id"] for r in result["high_priority"]], ["a", "b"])
        self.assertEqual([r["alert_id"] for r in result["low_priority"]], ["c"])
        self.assertEqual(result["input_count"], 3)
        json.dumps(result, allow_nan=False)

    def test_empty_window_is_not_counted_for_absent_alert(self):
        acc = self.accumulator()
        a = alert(1, alert_id="a")
        acc.add(self.window(0, [a], [(["a"], 0.5)]), [a])
        acc.add(self.window(60, [], [], threshold=None), [])
        result = acc.finalize()
        self.assertEqual(result["ranked"][0]["window_count"], 1)
        self.assertEqual(result["ranked"][0]["anomaly_score"], 0.5)
        self.assertEqual(result["high_priority"], [])
        self.assertEqual(self.accumulator().finalize()["ranked"], [])

    def test_duplicate_window_and_members_are_rejected(self):
        acc = self.accumulator()
        a = alert(1, alert_id="a")
        w = self.window(0, [a], [(["a"], 0.8)])
        acc.add(w, [a])
        with self.assertRaises(ValueError):
            acc.add(w, [a])
        with self.assertRaises(ValueError):
            acc.add(self.window(60, [a], [(["a", "a"], 0.7)]), [a])
        self.assertEqual(acc.finalize()["ranked"][0]["window_count"], 1)

    def test_pipeline_round_summaries_match_independent_window_average(self):
        pipeline = importlib.import_module("main")
        rows = [alert(0, alert_id="a"), alert(599, alert_id="b"),
                alert(600, host="other", alert_id="c"), alert(1800, alert_id="d")]
        results = list(pipeline.run_pipeline(rows, start_time=0, include_aggregates=True))
        summaries = [r for r in results if r["record_type"] == "round_summary"]
        windows = [r for r in results if r["record_type"] == "window"]
        self.assertEqual(len(windows), 42)
        self.assertEqual(len(summaries), 2)
        self.assertEqual(results[21]["record_type"], "round_summary")
        seen = []
        for summary in summaries:
            expected = {}
            for w in windows:
                if w["round_index"] == summary["round_index"]:
                    for group in w["priority_result"]["ranked"]:
                        for aid in group["member_ids"]:
                            expected.setdefault(aid, []).append(group["anomaly_score"])
            self.assertEqual(summary["available_at"], summary["round_end"])
            for row in summary["priority_result"]["ranked"]:
                scores = expected[row["alert_id"]]
                self.assertAlmostEqual(row["anomaly_score"], sum(scores) / len(scores))
                self.assertEqual(row["window_count"], len(scores))
                seen.append(row["alert_id"])
        self.assertCountEqual(seen, ["a", "b", "c", "d"])


class StormTests(unittest.TestCase):
    def pipeline(self):
        module = importlib.import_module("main")
        self.assertTrue(hasattr(module, "select_storm_periods"), "Storm selection is missing")
        return module

    def test_count_before_merge_and_labels_not_used(self):
        module = self.pipeline()
        rows = [alert(i, event_label="-") for i in range(3)]
        selected = list(module.select_storm_periods(rows, threshold=3, start_time=0))
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["trigger_count"], 3)
        self.assertEqual(selected[0]["trigger_time"], 600)
        for row in rows:
            row["event_label"] = "attack"
        self.assertEqual(selected, list(module.select_storm_periods(rows, threshold=3, start_time=0)))

    def test_boundary_and_nonoverlap(self):
        module = self.pipeline()
        rows = [alert(t) for t in [0, 599, 600, 1800, 1801]]
        selected = list(module.select_storm_periods(rows, threshold=2, start_time=0))
        self.assertEqual([s["round_start"] for s in selected], [0, 1800])
        self.assertEqual(selected[0]["trigger_count"], 2)
        self.assertEqual([s["round_end"] for s in selected], [1800, 3600])
        self.assertEqual(len(list(module.select_storm_periods(rows, threshold=2, start_time=0, max_rounds=1))), 1)

    def test_no_future_trigger_or_forced_positive(self):
        module = self.pipeline()
        rows = [alert(0), alert(700), alert(701)]
        selected = list(module.select_storm_periods(rows, threshold=2, start_time=0))
        self.assertEqual(selected[0]["round_start"], 120)
        self.assertEqual(selected[0]["trigger_time"], 720)
        self.assertEqual(list(module.select_storm_periods(rows, threshold=4)), [])
        self.assertEqual(list(module.select_storm_periods([], threshold=2)), [])
        for threshold in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                list(module.select_storm_periods(rows, threshold=threshold))

    def test_storm_pipeline_preserves_source_ids_and_averages(self):
        module = self.pipeline()
        rows = [alert(0, alert_id="outside"), alert(700, alert_id="b"), alert(701, alert_id="c")]
        records = list(module.run_storm_pipeline(rows, threshold=2, start_time=0))
        self.assertEqual(len(records), 22)
        summary = records[-1]
        self.assertEqual(summary["storm_trigger"]["trigger_count"], 2)
        self.assertEqual(summary["record_type"], "round_summary")
        self.assertCountEqual([r["alert_id"] for r in summary["priority_result"]["ranked"]], ["b", "c"])


class WindowTests(unittest.TestCase):
    def pipeline(self):
        self.assertIsNotNone(importlib.util.find_spec("main"), "Pipeline main.py is missing")
        return importlib.import_module("main")

    def test_half_open_windows_and_round_boundaries(self):
        pipeline = self.pipeline()
        rows = [alert(t) for t in [1800, 0, 600, 599, 1799, 3600]]
        before = copy.deepcopy(rows)
        windows = list(pipeline.iter_windows(rows, start_time=0))
        self.assertEqual(len(windows), 63)
        self.assertEqual([r["raw_time"] for r in windows[0]["alerts"]], [0, 599])
        self.assertEqual(windows[20]["window_start"], 1200)
        self.assertEqual(windows[20]["window_end"], 1800)
        self.assertEqual(windows[21]["window_start"], 1800)
        self.assertEqual(windows[-1]["window_end"], 5400)
        seen = {}
        for window in windows:
            for row in window["alerts"]:
                self.assertTrue(window["window_start"] <= row["raw_time"] < window["window_end"])
                seen.setdefault(row["raw_time"], set()).add(row["alert_id"])
        self.assertEqual(set(seen), {0, 599, 600, 1799, 1800, 3600})
        self.assertTrue(all(len(ids) == 1 for ids in seen.values()))
        self.assertEqual(rows, before)

    def test_one_round_limit_empty_input_and_duplicate_ids(self):
        pipeline = self.pipeline()
        windows = list(pipeline.iter_windows([alert(0), alert(1900)], start_time=0, max_rounds=1))
        self.assertEqual(len(windows), 21)
        self.assertTrue(all(w["window_end"] <= 1800 for w in windows))
        self.assertEqual(list(pipeline.iter_windows([])), [])
        with self.assertRaises(ValueError):
            list(pipeline.iter_windows([alert(0, alert_id="a"), alert(1, alert_id="a")]))

    def test_pipeline_calls_priority_on_each_window(self):
        pipeline = self.pipeline()
        rows = [alert(0, host="a"), alert(599, host="b"), alert(600, host="c")]
        results = list(pipeline.run_pipeline(rows, start_time=0, max_rounds=1))
        self.assertEqual(len(results), 21)
        for window in results:
            output = window["priority_result"]
            self.assertEqual(output["input_count"], window["alert_count"])
            for item in output["ranked"]:
                self.assertTrue(window["window_start"] <= item["alert"]["raw_time"] < window["window_end"])
            json.dumps(window, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
