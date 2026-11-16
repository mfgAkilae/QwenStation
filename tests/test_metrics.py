import unittest

from qwen_station import metrics

LABEL = '{engine="0",model_name="qwen38-gittensor-dflash"}'


def render(**values):
    """A /metrics body in the shape vLLM emits, including noise the parser must skip."""
    base = {
        "running": 0, "waiting": 0, "kv": 0.0, "prompt": 0, "gen": 0, "queries": 0, "hits": 0, "preempt": 0,
        "stop": 0, "length": 0, "ttft_sum": 0.0, "count": 0, "e2e_sum": 0.0, "prefill_sum": 0.0,
        "decode_sum": 0.0, "drafts": 0, "draft_tokens": 0, "accepted": 0, "pos": (),
    }
    base.update(values)
    lines = [
        "# HELP vllm:num_requests_running Number of requests in model execution batches.",
        "# TYPE vllm:num_requests_running gauge",
        f"vllm:num_requests_running{LABEL} {base['running']}.0",
        f"vllm:num_requests_waiting{LABEL} {base['waiting']}.0",
        f"vllm:kv_cache_usage_perc{LABEL} {base['kv']}",
        f"vllm:prompt_tokens_total{LABEL} {base['prompt']}.0",
        f"vllm:prompt_tokens_created{LABEL} 1.7948e+09",
        f"vllm:generation_tokens_total{LABEL} {base['gen']}.0",
        f"vllm:prefix_cache_queries_total{LABEL} {base['queries']}.0",
        f"vllm:prefix_cache_hits_total{LABEL} {base['hits']}.0",
        f"vllm:num_preemptions_total{LABEL} {base['preempt']}.0",
        f'vllm:request_success_total{{engine="0",finished_reason="stop",model_name="m"}} {base["stop"]}.0',
        f'vllm:request_success_total{{engine="0",finished_reason="length",model_name="m"}} {base["length"]}.0',
        f'vllm:time_to_first_token_seconds_bucket{{engine="0",le="0.5",model_name="m"}} 1.0',
        f"vllm:time_to_first_token_seconds_sum{LABEL} {base['ttft_sum']}",
        f"vllm:time_to_first_token_seconds_count{LABEL} {base['count']}.0",
        f"vllm:e2e_request_latency_seconds_sum{LABEL} {base['e2e_sum']}",
        f"vllm:e2e_request_latency_seconds_count{LABEL} {base['count']}.0",
        f"vllm:request_prefill_time_seconds_sum{LABEL} {base['prefill_sum']}",
        f"vllm:request_decode_time_seconds_sum{LABEL} {base['decode_sum']}",
        f"vllm:spec_decode_num_drafts_total{LABEL} {base['drafts']}.0",
        f"vllm:spec_decode_num_draft_tokens_total{LABEL} {base['draft_tokens']}.0",
        f"vllm:spec_decode_num_accepted_tokens_total{LABEL} {base['accepted']}.0",
        "process_cpu_seconds_total 12.5",
        "garbage line without value",
    ]
    for index, value in enumerate(base["pos"]):
        lines.append(f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",position="{index}"}} {value}.0')
    return "\n".join(lines) + "\n"


def snapshot(**values):
    return metrics.extract(metrics.parse_prometheus(render(**values)))


class ParserTests(unittest.TestCase):
    def test_parses_labels_values_and_skips_comments(self):
        samples = metrics.parse_prometheus(render(running=1, kv=0.4123, stop=3, length=1))
        self.assertEqual(samples["vllm:num_requests_running"][0][1], 1.0)
        self.assertEqual(samples["vllm:num_requests_running"][0][0]["model_name"], "qwen38-gittensor-dflash")
        self.assertNotIn("garbage", samples)
        reasons = {labels["finished_reason"]: value for labels, value in samples["vllm:request_success_total"]}
        self.assertEqual(reasons, {"stop": 3.0, "length": 1.0})

    def test_scientific_notation_nan_and_escaped_quotes(self):
        samples = metrics.parse_prometheus('a_total{x="q\\"uote"} 1.5e+03\nb_total NaN\nc_total 7 1700000000000\n')
        self.assertEqual(samples["a_total"], [({"x": 'q"uote'}, 1500.0)])
        self.assertNotIn("b_total", samples)
        self.assertEqual(samples["c_total"][0][1], 7.0)

    def test_extract_reads_the_families_and_tolerates_missing_ones(self):
        data = snapshot(running=1, kv=0.25, prompt=1000, gen=200, stop=2, drafts=10, accepted=42, pos=(9, 7, 5))
        self.assertEqual(data["requests"], 2.0)
        self.assertEqual(data["kv_usage"], 0.25)
        self.assertEqual(data["accepted_per_pos"], {0: 9.0, 1: 7.0, 2: 5.0})
        self.assertIsNone(data["cached_tokens"])  # this server build does not export it
        empty = metrics.extract(metrics.parse_prometheus(""))
        self.assertEqual(empty["generation_tokens"], 0.0)
        self.assertEqual(empty["accepted_per_pos"], {})

    def test_cached_tokens_from_either_family(self):
        direct = metrics.extract(metrics.parse_prometheus("vllm:prompt_tokens_cached_total 321\n"))
        self.assertEqual(direct["cached_tokens"], 321.0)
        by_source = metrics.extract(metrics.parse_prometheus(
            'vllm:prompt_tokens_by_source_total{source="local_cache_hit"} 200\n'
            'vllm:prompt_tokens_by_source_total{source="local_compute"} 50\n'))
        self.assertEqual(by_source["cached_tokens"], 200.0)


class TrackerTests(unittest.TestCase):
    def test_live_speed_activity_and_request_row(self):
        tracker = metrics.Tracker()
        tracker.update(snapshot(), now=100.0)
        prefill = tracker.update(snapshot(running=1), now=101.0)
        self.assertEqual(prefill["activity"], "prefill")
        self.assertEqual(prefill["tps"], 0.0)
        tracker.update(snapshot(running=1, prompt=5000, gen=150, kv=0.02), now=102.0)
        decode = tracker.update(snapshot(running=1, prompt=5000, gen=300, kv=0.02), now=103.0)
        self.assertEqual(decode["activity"], "decode")
        self.assertAlmostEqual(decode["tps"], 100.0)  # 300 tokens over the 3 s window
        self.assertEqual(len(tracker.requests), 0)
        done = tracker.update(snapshot(prompt=5000, gen=450, stop=1, ttft_sum=1.5, count=1, e2e_sum=4.5,
                                       prefill_sum=1.4, decode_sum=3.0, drafts=100, draft_tokens=700,
                                       accepted=350, queries=5000, hits=1000), now=104.0)
        self.assertEqual(done["activity"], "idle")
        row = tracker.requests[0]
        self.assertEqual((row["prompt_tokens"], row["output_tokens"], row["count"]), (5000, 450, 1))
        self.assertEqual(row["ttft_s"], 1.5)
        self.assertEqual(row["decode_tps"], 150.0)
        self.assertEqual(row["accept_rate"], 0.5)
        self.assertEqual(row["finish"], {"stop": 1})
        self.assertIsNone(row["cached_tokens"])
        totals = tracker.totals()
        self.assertEqual(totals["prefix_hit_rate"], 0.2)
        self.assertEqual(totals["accept_length"], 4.5)

    def test_second_request_only_counts_its_own_tokens(self):
        tracker = metrics.Tracker()
        tracker.update(snapshot(prompt=100, gen=50, stop=1, count=1, ttft_sum=1.0), now=10.0)
        tracker.update(snapshot(running=1, prompt=400, gen=80, stop=1, count=1, ttft_sum=1.0), now=11.0)
        tracker.update(snapshot(prompt=400, gen=170, stop=1, length=1, count=2, ttft_sum=3.0, decode_sum=2.0), now=12.0)
        row = tracker.requests[0]
        self.assertEqual((row["prompt_tokens"], row["output_tokens"]), (300, 120))
        self.assertEqual(row["ttft_s"], 2.0)
        self.assertEqual(row["finish"], {"length": 1})

    def test_aborted_request_gets_its_own_row(self):
        tracker = metrics.Tracker()
        tracker.update(snapshot(), now=1.0)
        tracker.update(snapshot(running=1, prompt=900, gen=40), now=2.0)
        tracker.update(snapshot(prompt=900, gen=40), now=3.0)  # running fell to 0, nothing counted as finished
        self.assertEqual(tracker.requests[0]["finish"], {"abort": 1})
        self.assertEqual(tracker.requests[0]["output_tokens"], 40)

    def test_server_restart_resets_the_baseline(self):
        tracker = metrics.Tracker()
        tracker.update(snapshot(prompt=9000, gen=5000, stop=7, count=7), now=1.0)
        after = tracker.update(snapshot(prompt=10, gen=5), now=2.0)  # counters began again
        self.assertEqual(after["tps"], 0.0)
        self.assertEqual(len(tracker.requests), 0)

    def test_per_position_rates_and_empty_totals(self):
        tracker = metrics.Tracker()
        self.assertIsNone(tracker.totals())
        tracker.update(snapshot(drafts=100, pos=(87, 74, 62)), now=1.0)
        self.assertEqual(tracker.totals()["accept_per_pos"], [0.87, 0.74, 0.62])
        self.assertIsNone(tracker.totals()["prefix_hit_rate"])


if __name__ == "__main__":
    unittest.main()
