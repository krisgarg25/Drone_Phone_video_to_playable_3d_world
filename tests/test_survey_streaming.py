"""CPU tests for the streaming/throughput model.

Nothing here measures a real run: no 600-second single-pass video exists on this
machine. The model's job is to turn the per-image rates that *were* measured
(report section 9.1) into an explicit prediction that can say "infeasible".
"""
import json
import unittest

from scripts import survey_streaming as st


def _rates(**overrides):
    rates = {name: dict(spec) for name, spec in st.MEASURED_RATES.items()}
    rates.update(overrides)
    return rates


class StageBudgetTests(unittest.TestCase):
    def test_measured_rates_reproduce_the_reported_4001_second_run(self):
        budget = st.plan_stages(291, 1000, rates=st.MEASURED_RATES)
        self.assertAlmostEqual(budget.stages["keyframes"], 21.5, places=2)
        self.assertAlmostEqual(budget.stages["sparse"], 946.9, places=2)
        self.assertAlmostEqual(budget.stages["undistort"], 16.3, places=2)
        self.assertAlmostEqual(budget.stages["dense"], 2951.8, places=1)
        self.assertAlmostEqual(budget.stages["fusion"], 64.9, places=2)
        self.assertAlmostEqual(budget.total_s, 4001.4, places=1)
        self.assertEqual(budget.frame_count, 291)
        self.assertEqual(budget.image_px, 1000)
        self.assertEqual(budget.label, st.PREDICTED_LABEL)
        stages = budget.to_dict()["stages"]
        self.assertEqual(list(stages), ["keyframes", "sparse", "undistort", "dense", "fusion"])
        for name, seconds in budget.stages.items():
            self.assertAlmostEqual(stages[name]["seconds"], seconds, places=6, msg=name)
            self.assertEqual(stages[name]["status"], st.PREDICTED_LABEL)

    def test_measured_dense_budget_is_reported_infeasible_inside_900_s(self):
        budget = st.plan_stages(291, 1000, rates=st.MEASURED_RATES, deadline_s=900.0)
        self.assertFalse(budget.fits_deadline)
        self.assertGreater(budget.stages["dense"], 900.0)  # 2951.8 s on its own
        self.assertAlmostEqual(budget.overage_s, budget.total_s - 900.0, places=6)
        self.assertGreater(budget.processing_ratio, 6.0)   # 4001.4 / 600
        self.assertFalse(budget.meets_ratio_target)
        self.assertFalse(budget.overage_s <= 0.0)
        payload = json.dumps(budget.to_dict())
        self.assertIn(st.PREDICTED_LABEL, payload)
        self.assertNotIn("meets the target", payload)

    def test_negative_headroom_is_reported_as_zero_overage(self):
        budget = st.plan_stages(24, 800, rates=st.MEASURED_RATES, deadline_s=900.0)
        self.assertTrue(budget.fits_deadline)
        self.assertEqual(budget.overage_s, 0.0)
        self.assertGreater(budget.headroom_s, 600.0)

    def test_deadline_is_a_parameter(self):
        self.assertTrue(st.plan_stages(291, 1000, rates=st.MEASURED_RATES,
                                       deadline_s=4002.0).fits_deadline)
        self.assertFalse(st.plan_stages(291, 1000, rates=st.MEASURED_RATES,
                                        deadline_s=899.99).fits_deadline)
        with self.assertRaises(ValueError):
            st.plan_stages(291, 1000, rates=st.MEASURED_RATES, deadline_s=0)

    def test_resolution_scaling_only_happens_when_the_rate_declares_its_px(self):
        scaled = st.plan_stages(10, 2000, rates=st.MEASURED_RATES)
        base = st.plan_stages(10, 1000, rates=st.MEASURED_RATES)
        self.assertAlmostEqual(scaled.stages["dense"], 2.0 * base.stages["dense"], places=6)
        # keyframes declares pixel_power 0.0: decoding does not care about the
        # working resolution, so the model must not silently scale it.
        self.assertAlmostEqual(scaled.stages["keyframes"], base.stages["keyframes"], places=6)
        unanchored = _rates(dense={"model": "per_image", "s_per_image": 10.0})
        self.assertAlmostEqual(st.plan_stages(10, 4000, rates=unanchored).stages["dense"],
                               100.0, places=6)

    def test_pairwise_stages_scale_with_pairs_not_images(self):
        sequential = {"matching": {"model": "pairs",
                                   "s_per_pair": st.MEASURED_SEQUENTIAL_MATCHING["s_per_pair"],
                                   "pair_strategy": "sequential", "reference_px": 1000}}
        budget = st.plan_stages(291, 1000, rates=sequential)
        # The measured sequential-only matching run: 239.7 s at 291 frames, overlap 20.
        self.assertAlmostEqual(budget.stages["matching"], 239.7, places=2)
        self.assertEqual(budget.to_dict()["bases"]["matching"]["pairs"], 291 * 20)
        exhaustive = {"matching": {"model": "pairs", "s_per_pair": 0.0411856,
                                   "pair_strategy": "exhaustive"}}
        double = st.plan_stages(2 * 291, 1000, rates=exhaustive)
        single = st.plan_stages(291, 1000, rates=exhaustive)
        self.assertAlmostEqual(single.stages["matching"], 0.0411856 * 291 * 290 / 2.0, places=4)
        self.assertGreater(double.stages["matching"], 3.9 * single.stages["matching"])
        neighbour = st.plan_stages(291, 1000, rates=sequential, match_neighbors=10)
        self.assertAlmostEqual(neighbour.stages["matching"],
                               budget.stages["matching"] / 2.0, places=6)
        # A pair count can never exceed the pairs that exist.
        tiny = st.plan_stages(3, 1000, rates=sequential)
        self.assertEqual(tiny.to_dict()["bases"]["matching"]["pairs"], 3 * 2)

    def test_fixed_stages_do_not_move_with_frames(self):
        rates = _rates(vocab_tree={"model": "fixed", "fixed_s": 12.0})
        small = st.plan_stages(10, 1000, rates=rates)
        big = st.plan_stages(200, 1000, rates=rates)
        self.assertEqual(small.stages["vocab_tree"], 12.0)
        self.assertEqual(big.stages["vocab_tree"], 12.0)
        self.assertGreater(big.stages["dense"], small.stages["dense"])

    def test_product_tiers_are_declared_by_the_caller(self):
        rates = {"poses": {"model": "per_image", "s_per_image": 0.01, "product": "coarse"},
                 "dense": {"model": "per_image", "s_per_image": 1.0, "product": "final"}}
        budget = st.plan_stages(100, 1000, rates=rates)
        self.assertEqual(budget.coarse_s, 1.0)
        self.assertAlmostEqual(budget.final_s, budget.total_s)
        self.assertAlmostEqual(budget.total_s, 101.0, places=6)
        none = st.plan_stages(100, 1000, rates={"dense": {"model": "per_image",
                                                          "s_per_image": 1.0}})
        self.assertIsNone(none.coarse_s)
        self.assertEqual(none.final_s, none.total_s)

    def test_rate_table_is_validated(self):
        bad_tables = (
            {},                                            # nothing to plan
            {"dense": 10.1},                               # not a spec
            {"dense": {"s_per_image": 10.1}},              # missing model
            {"dense": {"model": "per_image"}},             # missing rate
            {"dense": {"model": "per_image", "s_per_image": -1.0}},
            {"dense": {"model": "per_image", "s_per_image": 1.0, "mystery": 2}},
            {"dense": {"model": "per_image", "s_per_image": 1.0, "reference_px": 0}},
            {"dense": {"model": "pairs", "fixed_s": 1.0}},
            {"dense": {"model": "per_image", "s_per_image": 1.0, "pair_strategy": "sequential"}},
            {"dense": {"model": "pairs", "s_per_pair": 1.0, "pair_strategy": "random"}},
            {"dense": {"model": "nonsense", "s_per_image": 1.0}},
            {"dense": {"model": "per_image", "s_per_image": float("nan")}},
            {"dense": {"model": "per_image", "s_per_image": 1.0, "product": "maybe"}},
        )
        for table in bad_tables:
            with self.subTest(table=table), self.assertRaises(ValueError):
                st.plan_stages(10, 1000, rates=table)
        for args in ((0, 1000), (-3, 1000), (10, 0), (10, -1), (10, 1000.5), (2.5, 1000)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                st.plan_stages(*args, rates=st.MEASURED_RATES)
        with self.assertRaises(ValueError):
            st.plan_stages(10, 1000, rates=st.MEASURED_RATES, match_neighbors=0)

    def test_inputs_are_never_mutated(self):
        rates = _rates()
        before = json.dumps(rates, sort_keys=True)
        st.plan_stages(291, 1000, rates=rates)
        self.assertEqual(json.dumps(rates, sort_keys=True), before)


class BudgetSearchTests(unittest.TestCase):
    OPTIONS = {"frame_options": (291, 120, 69, 40, 24), "resolution_options": (1000, 800)}

    def test_291_frames_is_infeasible_and_the_model_says_so(self):
        result = st.search_budget(st.MEASURED_RATES, 600.0, 900.0,
                                  frame_options=(291,), resolution_options=(1000, 2000))
        self.assertEqual(result["status"], "infeasible")
        self.assertIsNone(result["config"])
        self.assertIsNone(result["predicted_s"])
        self.assertIn("unreachable", result["reason"])
        self.assertIn("291", result["reason"])
        self.assertIn("900", result["reason"])
        self.assertEqual(result["label"], st.PREDICTED_LABEL)
        self.assertEqual(result["candidates_evaluated"], 2)
        self.assertAlmostEqual(result["fastest_predicted_s"], 4001.4, places=1)
        self.assertEqual(result["feasible"], [])

    def test_a_reduced_frame_configuration_is_feasible(self):
        result = st.search_budget(st.MEASURED_RATES, 600.0, 900.0, **self.OPTIONS)
        self.assertEqual(result["status"], "feasible")
        self.assertEqual(result["config"], {"frame_count": 24, "image_px": 800})
        self.assertLess(result["predicted_s"], 900.0)
        self.assertAlmostEqual(result["predicted_s"],
                               st.plan_stages(24, 800, rates=st.MEASURED_RATES).total_s,
                               places=6)
        # The most frames that still fit are the 69 the baseline-aware selector
        # measured - and only once the working resolution drops to 800 px.
        self.assertEqual(result["best_quality_within_deadline"]["frame_count"], 69)
        self.assertEqual(result["best_quality_within_deadline"]["image_px"], 800)
        self.assertLess(result["best_quality_predicted_s"], 900.0)
        self.assertTrue(all(c["predicted_s"] < 900.0 for c in result["feasible"]))
        self.assertEqual(len(result["feasible"]), 5)   # 69@800, 40 and 24 at both sizes
        self.assertEqual(len(result["infeasible"]), 5)

    def test_frame_supply_cap_is_enforced_and_explained(self):
        result = st.search_budget(st.MEASURED_RATES, 600.0, 900.0,
                                  frame_options=(291, 120), resolution_options=(1000,),
                                  max_frame_rate=0.1)  # 1 frame per 10 s -> 60 frames
        self.assertEqual(result["status"], "infeasible")
        self.assertIn("frame supply", result["reason"])
        self.assertEqual(result["candidates_rejected_for_frame_supply"], 2)
        self.assertIsNone(result["fastest_predicted_s"])

    def test_search_rejects_nonsense_options(self):
        with self.assertRaises(ValueError):
            st.search_budget(st.MEASURED_RATES, 600.0, 900.0, frame_options=(),
                             resolution_options=(1000,))
        with self.assertRaises(ValueError):
            st.search_budget(st.MEASURED_RATES, 600.0, 900.0, frame_options=291,
                             resolution_options=(1000,))
        with self.assertRaises(ValueError):
            st.search_budget(st.MEASURED_RATES, 0.0, 900.0, frame_options=(60,),
                             resolution_options=(1000,))
        with self.assertRaises(ValueError):
            st.search_budget(st.MEASURED_RATES, 600.0, 900.0, frame_options=(-1,),
                             resolution_options=(1000,))
        with self.assertRaises(ValueError):
            st.search_budget(st.MEASURED_RATES, 600.0, 0.0, frame_options=(60,),
                             resolution_options=(1000,))


class ProgressivePlanTests(unittest.TestCase):
    def test_windows_are_ordered_with_the_requested_overlap(self):
        plan = st.progressive_plan(291, 4)
        self.assertEqual(plan["windows_declared"], 4)
        self.assertEqual(len(plan["windows"]), 4)
        self.assertGreaterEqual(plan["overlap_frames"], plan["min_overlap_frames"])
        # The overlap is a fraction of each window's *new* frames, which is what
        # registration actually needs: shared views relative to the views added.
        self.assertEqual(plan["stride_frames"] + plan["overlap_frames"],
                         plan["window_frames"])
        self.assertGreaterEqual(plan["overlap_frames"],
                                plan["min_overlap_fraction"] * plan["stride_frames"])
        first, last = plan["windows"][0], plan["windows"][-1]
        self.assertEqual(first["start"], 0)
        self.assertEqual(last["end"], 291)
        self.assertTrue(plan["coverage_complete"])
        for window, previous in zip(plan["windows"][1:], plan["windows"][:-1]):
            self.assertEqual(window["start"], previous["end"] - plan["overlap_frames"])
            self.assertEqual(window["overlap_with_previous"], plan["overlap_frames"])
        for window in plan["windows"]:
            self.assertGreaterEqual(window["frame_count"], plan["min_window_frames"])
            self.assertEqual(window["frame_count"], window["end"] - window["start"])
        self.assertEqual(plan["windows"][0]["overlap_with_previous"], 0)
        self.assertEqual(plan["frames_processed"], sum(w["frame_count"] for w in
                                                      plan["windows"]))
        self.assertGreater(plan["duplicate_frames"], 0)
        self.assertAlmostEqual(plan["duplicate_fraction"],
                               plan["duplicate_frames"] / plan["frames_processed"], places=6)
        self.assertEqual(plan["label"], st.PREDICTED_LABEL)

    def test_more_windows_cost_more_duplicated_work(self):
        few = st.progressive_plan(291, 2)
        many = st.progressive_plan(291, 12)
        self.assertGreater(many["duplicate_fraction"], few["duplicate_fraction"])
        self.assertLess(many["window_frames"], few["window_frames"])

    def test_times_are_attached_only_when_rates_are_supplied(self):
        bare = st.progressive_plan(291, 4)
        self.assertIsNone(bare["windows"][0]["predicted_s"])
        timed = st.progressive_plan(291, 4, rates=st.MEASURED_RATES, image_px=1000)
        seconds = [w["predicted_s"] for w in timed["windows"]]
        cumulative = [w["cumulative_s"] for w in timed["windows"]]
        self.assertTrue(all(s is not None for s in seconds))
        self.assertTrue(all(b > a for a, b in zip(cumulative, cumulative[1:])))
        self.assertAlmostEqual(cumulative[-1], sum(seconds), places=4)
        # Windowing is not free: the overlap is reconstructed twice.
        self.assertGreater(cumulative[-1], st.plan_stages(291, 1000,
                                                          rates=st.MEASURED_RATES).total_s)
        # Even the first window's dense stage busts the 900 s gate at this size.
        self.assertGreater(cumulative[0], st.DEFAULT_DEADLINE_S)

    def test_progressive_inputs_are_validated(self):
        for windows in (0, 1, 1000):
            with self.subTest(windows=windows), self.assertRaises(ValueError):
                st.progressive_plan(291, windows)
        with self.assertRaises(ValueError):
            st.progressive_plan(291, 4, overlap_frames=0)
        with self.assertRaises(ValueError):
            st.progressive_plan(291, 4, overlap_frames=290)
        with self.assertRaises(ValueError):
            st.progressive_plan(0, 4)
        with self.assertRaises(ValueError):
            st.progressive_plan(291, 4, rates=st.MEASURED_RATES)  # no image_px


class FirstOutputTests(unittest.TestCase):
    def test_coarse_and_final_products_are_kept_apart(self):
        output = st.first_useful_output_at(291, 1000, rates=st.MEASURED_RATES, windows=4)
        self.assertEqual(output["label"], st.PREDICTED_LABEL)
        self.assertAlmostEqual(output["coarse_ready_s"], 968.4, places=1)
        self.assertAlmostEqual(output["final_ready_s"], 4001.4, places=1)
        self.assertLess(output["first_measurable_map_s"], output["coarse_ready_s"])
        self.assertLess(output["coarse_ready_s"], output["final_ready_s"])
        # Dense stereo is why the final product misses; the first submap does not.
        self.assertGreater(output["coarse_ready_s"], 900.0)
        self.assertLess(output["first_window_coarse_ready_s"], 900.0)
        self.assertTrue(output["meets_deadline_first_map"])
        self.assertFalse(output["meets_deadline_final"])
        self.assertIn("submap", output["statement"])
        self.assertIn("final", output["statement"])

    def test_final_only_rate_table_reports_no_coarse_product(self):
        rates = {"dense": {"model": "per_image", "s_per_image": 10.143643}}
        output = st.first_useful_output_at(20, 1000, rates=rates, windows=2)
        self.assertIsNone(output["coarse_ready_s"])
        self.assertIsNone(output["first_measurable_map_s"])
        self.assertAlmostEqual(output["final_ready_s"], 202.87, places=2)
        self.assertFalse(output["meets_deadline_first_map"])
        self.assertIn("No coarse-product stages", output["statement"])
        self.assertIn("no early measurable map", output["statement"])

    def test_first_output_inputs_are_validated(self):
        with self.assertRaises(ValueError):
            st.first_useful_output_at(291, 1000, rates=st.MEASURED_RATES, windows=0)
        with self.assertRaises(ValueError):
            st.first_useful_output_at(291, 1000, rates=st.MEASURED_RATES, deadline_s=-5)


class ReportTests(unittest.TestCase):
    def test_report_labels_every_number_as_a_prediction(self):
        payload = st.report(st.MEASURED_RATES, 291, 1000, windows=4,
                            frame_options=(291, 69, 24), resolution_options=(1000, 800))
        text = json.dumps(payload)
        self.assertIn(st.PREDICTED_LABEL, text)
        self.assertIn(st.NOT_MEASURED_LABEL, text)
        # "measured end-to-end" may only ever appear negated.
        self.assertEqual(text.count("measured end-to-end"),
                         text.count("not measured end-to-end"))
        self.assertEqual(payload["status_label"], st.PREDICTED_LABEL)
        self.assertEqual(payload["budget"]["label"], st.PREDICTED_LABEL)
        for stage in payload["budget"]["stages"].values():
            self.assertEqual(stage["status"], st.PREDICTED_LABEL)
        self.assertEqual(payload["provenance"]["clip"], "room_w_jsonl")
        self.assertEqual(payload["provenance"]["keyframes"], 291)
        self.assertEqual(payload["provenance"]["image_px"], 1000)
        self.assertEqual(payload["search"]["status"], "feasible")
        self.assertTrue(any("600" in line for line in payload["disclaimers"]),
                        payload["disclaimers"])

    def test_report_states_the_stage_that_domines_the_budget(self):
        payload = st.report(st.MEASURED_RATES, 291, 1000)
        self.assertEqual(payload["dominant_stage"], "dense")
        self.assertGreater(payload["dominant_stage_fraction"], 0.7)
        self.assertAlmostEqual(payload["budget"]["total_s"], 4001.4, places=1)
        self.assertFalse(payload["budget"]["fits_deadline"])
        self.assertIsNone(payload["search"])
        self.assertIn("dense", json.dumps(payload["statement"]))
        self.assertEqual(payload["deadline_s"], st.DEFAULT_DEADLINE_S)
        self.assertEqual(payload["video_duration_s"], st.DEFAULT_VIDEO_DURATION_S)

    def test_report_rejects_an_unusable_rate_table(self):
        with self.assertRaises(ValueError):
            st.report({}, 291, 1000)
        with self.assertRaises(ValueError):
            st.report(st.MEASURED_RATES, 291, 1000, deadline_s=-1)


if __name__ == "__main__":
    unittest.main()
