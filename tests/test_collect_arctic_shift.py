import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

from src.collect_arctic_shift_core import (
    AGE_ASSURANCE_QUERIES,
    API_BASE,
    API_DOCS_URL,
    COMMENT_TREE_LIMIT,
    EVENT_WINDOWS,
    MalformedApiResponseError,
    STUDY_NAME,
    SUBREDDIT_SCOPE,
    TARGET_SUBREDDITS,
    ApiTimeoutError,
    SearchTask,
    Stats,
    TaskResult,
    RequestBudgetExceeded,
    RELEVANCE_SCREENING,
    build_plan_summary,
    collect_comment_tree,
    collect_interval,
    compute_completion_status,
    consolidate_acquisition_matches,
    execute_planned_tasks,
    expected_task_count,
    flatten_comment_tree,
    iso_utc,
    is_timeout_message,
    load_matched_post_targets,
    merge_task_history,
    parse_utc,
    request_json,
    response_rows,
    run_collection,
    split_window,
    task_identity,
    task_collection_base,
    write_manifest,
    validate_resume_manifest,
    write_rows,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.payload


def collector_args(**overrides):
    values = {
        "max_requests": None,
        "delay": 0,
        "max_retries": 0,
        "min_window_seconds": 2,
        "comment_tree_limit": COMMENT_TREE_LIMIT,
        "limit": 100,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def event_task(query="age verification"):
    event_id, event = next(iter(EVENT_WINDOWS.items()))
    return SearchTask(
        subreddit="privacy",
        event_id=event_id,
        jurisdiction=event["jurisdiction"],
        window_start_utc=event["start_utc"],
        window_end_utc=event["end_utc"],
        query=query,
    )


class CollectorCoreTests(unittest.TestCase):
    def test_plan_covers_all_configured_communities_events_and_queries(self):
        tasks, count = build_plan_summary()
        self.assertEqual(TARGET_SUBREDDITS, ["privacy", "technology", "parenting", "unitedkingdom", "europe", "australia"])
        self.assertNotIn("vietnam", TARGET_SUBREDDITS)
        self.assertEqual(count, 3 * 6 * 15)
        self.assertEqual(len(tasks), 270)
        self.assertEqual({task.event_id for task in tasks}, set(EVENT_WINDOWS))
        self.assertEqual({task.query for task in tasks}, set(AGE_ASSURANCE_QUERIES))

    def test_timestamps_are_round_trip_utc(self):
        value = "2024-11-10T00:00:00Z"
        self.assertEqual(iso_utc(parse_utc(value)), value)

    def test_malformed_response_fails_closed(self):
        with self.assertRaises(MalformedApiResponseError):
            response_rows({})
        with self.assertRaises(MalformedApiResponseError):
            response_rows({"data": None})

    def test_timeout_payload_and_documented_http_timeout_are_detected(self):
        self.assertTrue(is_timeout_message("Query timed out"))
        with patch("src.collect_arctic_shift_core.urlopen", return_value=FakeResponse({"error": "timeout"})):
            with self.assertRaises(ApiTimeoutError):
                request_json("https://example.test", delay=0, max_retries=0, stats=Stats())

        body = io.BytesIO(b'{"error":"query timed out"}')
        error = HTTPError("https://example.test", 422, "Unprocessable Entity", {}, body)
        with patch("src.collect_arctic_shift_core.urlopen", side_effect=error):
            with self.assertRaises(ApiTimeoutError):
                request_json("https://example.test", delay=0, max_retries=0, stats=Stats())
        self.assertTrue(body.closed)

    def test_http_timeout_gets_one_transient_retry_when_enabled(self):
        body = io.BytesIO(b'{"error":"query timed out"}')
        error = HTTPError("https://example.test", 422, "Unprocessable Entity", {}, body)
        stats = Stats()
        with patch("src.collect_arctic_shift_core.urlopen", side_effect=[error, FakeResponse({"data": []})]) as request:
            with patch("src.collect_arctic_shift_core.time.sleep") as sleep:
                response = request_json("https://example.test", delay=0, max_retries=2, stats=stats)
        self.assertEqual(response, {"data": []})
        self.assertEqual(request.call_count, 2)
        self.assertEqual(stats.retries, 1)
        self.assertEqual(sleep.call_count, 1)

    def test_json_timeout_gets_one_transient_retry_when_enabled(self):
        stats = Stats()
        with patch(
            "src.collect_arctic_shift_core.urlopen",
            side_effect=[FakeResponse({"error": "timeout"}), FakeResponse({"data": []})],
        ) as request:
            with patch("src.collect_arctic_shift_core.time.sleep") as sleep:
                response = request_json("https://example.test", delay=0, max_retries=2, stats=stats)
        self.assertEqual(response, {"data": []})
        self.assertEqual(request.call_count, 2)
        self.assertEqual(stats.retries, 1)
        self.assertEqual(sleep.call_count, 1)

    def test_request_budget_applies_to_each_retry_attempt(self):
        body = io.BytesIO(b'{"error":"temporary"}')
        error = HTTPError("https://example.test", 500, "Server Error", {}, body)
        stats = Stats()
        with patch("src.collect_arctic_shift_core.urlopen", side_effect=error) as request:
            with patch("src.collect_arctic_shift_core.time.sleep"):
                with self.assertRaises(RequestBudgetExceeded):
                    request_json(
                        "https://example.test",
                        delay=0,
                        max_retries=2,
                        stats=stats,
                        max_requests=1,
                    )
        self.assertEqual(request.call_count, 1)
        self.assertEqual(stats.requests, 1)
        self.assertEqual(stats.run_requests, 1)

    def test_split_makes_progress(self):
        self.assertIsNone(split_window(100, 102))
        left, right = split_window(100, 103)
        self.assertLessEqual(right[0], left[1])
        self.assertLess(left[1] - left[0], 3)
        self.assertLess(right[1] - right[0], 3)

    def test_post_search_filters_rows_to_exact_event_window(self):
        task = event_task()
        inside = {"id": "inside", "created_utc": task.start_epoch}
        outside = {"id": "outside", "created_utc": task.start_epoch - 1}
        args = collector_args()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "posts_raw.jsonl"
            stats = Stats()
            with patch("src.collect_arctic_shift_core.request_json", return_value={"data": [inside, outside]}):
                result = collect_interval(
                    endpoint="/api/posts/search",
                    base_params={"subreddit": task.subreddit, "query": task.query},
                    task=task,
                    start_epoch=task.start_epoch,
                    end_epoch=task.end_epoch,
                    limit=100,
                    min_window_seconds=2,
                    out_file=output,
                    seen=set(),
                    collection_base=task_collection_base(task, task.query),
                    stats=stats,
                    args=args,
                )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(result.status, "complete")
        self.assertEqual([row["data"]["id"] for row in rows], ["t3_inside"])
        self.assertEqual(stats.out_of_window_rows, 1)
        self.assertEqual(rows[0]["collection"]["event_id"], task.event_id)

    def test_api_provenance_records_the_actual_guarded_after_bound(self):
        task = event_task()
        captured = {}

        def fake_request(url, delay, max_retries, stats, max_requests=None):
            captured["url"] = url
            return {"data": [{"id": "post", "created_utc": task.start_epoch}]}

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "posts_raw.jsonl"
            with patch("src.collect_arctic_shift_core.request_json", side_effect=fake_request):
                result = collect_interval(
                    endpoint="/api/posts/search",
                    base_params={"subreddit": task.subreddit, "query": task.query},
                    task=task,
                    start_epoch=task.start_epoch,
                    end_epoch=task.end_epoch,
                    limit=100,
                    min_window_seconds=2,
                    out_file=output,
                    seen=set(),
                    collection_base=task_collection_base(task, task.query),
                    stats=Stats(),
                    args=collector_args(),
                )
            wrapper = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result.status, "complete")
        query = parse_qs(urlsplit(captured["url"]).query)
        self.assertEqual(int(query["after"][0]), task.start_epoch - 1)
        self.assertEqual(wrapper["collection"]["api_params"]["after"], task.start_epoch - 1)

    def test_flattening_removes_nested_replies_and_reports_collapsed_or_unknown_nodes(self):
        comments, collapsed, unknown = flatten_comment_tree(
            [
                {
                    "kind": "t1",
                    "data": {
                        "id": "root",
                        "replies": {
                            "kind": "Listing",
                            "data": {
                                "children": [
                                    {"kind": "t1", "data": {"id": "child", "replies": ""}},
                                    {"kind": "more", "data": {"id": "hidden", "count": 3}},
                                ]
                            },
                        },
                    },
                },
                {"kind": "future_node", "data": {}},
            ]
        )
        self.assertEqual([row["id"] for row in comments], ["root", "child"])
        self.assertNotIn("replies", comments[0])
        self.assertEqual(collapsed[0]["count"], 3)
        self.assertEqual(unknown[0]["kind"], "future_node")

    def test_comment_tree_filters_dates_preserves_network_ids_and_writes_flat_rows(self):
        event_id, event = next(iter(EVENT_WINDOWS.items()))
        target = {
            "post_id": "t3_post",
            "subreddit": "technology",
            "event_id": event_id,
            "jurisdiction": event["jurisdiction"],
            "window_start_utc": event["start_utc"],
            "window_end_utc": event["end_utc"],
            "queries": ["age verification", "digital ID"],
        }
        start = parse_utc(event["start_utc"])
        response = {
            "data": [
                {
                    "kind": "t1",
                    "data": {
                        "id": "root",
                        "created_utc": start,
                        "link_id": "t3_post",
                        "parent_id": "t3_post",
                        "body": "root",
                        "replies": {
                            "kind": "Listing",
                            "data": {
                                "children": [
                                    {
                                        "kind": "t1",
                                        "data": {
                                            "id": "child",
                                            "created_utc": start + 1,
                                            "link_id": "t3_post",
                                            "parent_id": "t1_root",
                                            "body": "child",
                                        },
                                    }
                                ]
                            },
                        },
                    },
                },
                {
                    "kind": "t1",
                    "data": {
                        "id": "outside",
                        "created_utc": parse_utc(event["end_utc"]),
                        "link_id": "t3_post",
                        "parent_id": "t3_post",
                        "body": "outside",
                    },
                },
            ]
        }
        args = collector_args()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "comments_raw.jsonl"
            routes = Path(tmp) / "acquisition_routes.jsonl"
            stats = Stats()
            with patch("src.collect_arctic_shift_core.request_json", return_value=response):
                task = collect_comment_tree(
                    target=target,
                    out_file=output,
                    seen=set(),
                    stats=stats,
                    args=args,
                    acquisition_file=routes,
                    route_seen=set(),
                )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            route_count = len(routes.read_text(encoding="utf-8").splitlines())

        self.assertEqual(task.status, "complete")
        self.assertEqual(task.rows_filtered_outside_window, 1)
        self.assertEqual([row["data"]["id"] for row in rows], ["t1_root", "t1_child"])
        self.assertEqual(rows[1]["data"]["parent_id"], "t1_root")
        self.assertEqual(rows[0]["data"]["parent_thing"], "post")
        self.assertNotIn("replies", rows[0]["data"])
        self.assertEqual(route_count, 4)

    def test_comment_tree_missing_relation_id_is_partial(self):
        event_id, event = next(iter(EVENT_WINDOWS.items()))
        target = {
            "post_id": "t3_post",
            "subreddit": "privacy",
            "event_id": event_id,
            "jurisdiction": event["jurisdiction"],
            "window_start_utc": event["start_utc"],
            "window_end_utc": event["end_utc"],
            "queries": ["age verification"],
        }
        response = {"data": [{"kind": "t1", "data": {"id": "bad", "created_utc": parse_utc(event["start_utc"]), "body": "bad"}}]}
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.collect_arctic_shift_core.request_json", return_value=response):
                task = collect_comment_tree(
                    target=target,
                    out_file=Path(tmp) / "comments_raw.jsonl",
                    seen=set(),
                    stats=Stats(),
                    args=collector_args(),
                )
        self.assertEqual(task.status, "partial")
        self.assertIn("missing usable ids", task.error)

    def test_unknown_tree_shape_is_partial(self):
        event_id, event = next(iter(EVENT_WINDOWS.items()))
        target = {
            "post_id": "t3_post",
            "subreddit": "privacy",
            "event_id": event_id,
            "jurisdiction": event["jurisdiction"],
            "window_start_utc": event["start_utc"],
            "window_end_utc": event["end_utc"],
            "queries": ["age verification"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            stats = Stats()
            with patch("src.collect_arctic_shift_core.request_json", return_value={"data": [{"kind": "future", "data": {}}]}):
                task = collect_comment_tree(
                    target=target,
                    out_file=Path(tmp) / "comments_raw.jsonl",
                    seen=set(),
                    stats=stats,
                    args=collector_args(),
                )
        self.assertEqual(task.status, "partial")
        self.assertEqual(task.unknown_tree_nodes, 1)
        self.assertEqual(stats.errors, 1)

    def test_duplicate_post_keeps_all_event_query_routes_and_target_queries(self):
        event_id, event = next(iter(EVENT_WINDOWS.items()))
        collection = {
            "platform": "reddit",
            "source_corpus": "reddit",
            "subreddit": "technology",
            "subreddit_scope": SUBREDDIT_SCOPE,
            "event_id": event_id,
            "jurisdiction": event["jurisdiction"],
            "window_start_utc": event["start_utc"],
            "window_end_utc": event["end_utc"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "posts_raw.jsonl"
            routes = Path(tmp) / "acquisition_routes.jsonl"
            seen, route_seen, stats = set(), set(), Stats()
            for query in ("age verification", "digital ID"):
                write_rows(
                    rows=[{"id": "post", "created_utc": parse_utc(event["start_utc"])}],
                    thing="post",
                    out_file=raw,
                    seen=seen,
                    collection={**collection, "query": query},
                    stats=stats,
                    acquisition_file=routes,
                    route_seen=route_seen,
                )
            consolidate_acquisition_matches(raw, routes)
            wrapper = json.loads(raw.read_text(encoding="utf-8"))
            targets = load_matched_post_targets(raw, routes)

        self.assertEqual(stats.rows_written, 1)
        self.assertEqual(stats.duplicates, 1)
        self.assertEqual(wrapper["data"]["id"], "t3_post")
        self.assertEqual(set(wrapper["collection"]["matched_queries"]), {"age verification", "digital ID"})
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["queries"], ["age verification", "digital ID"])

    def test_resume_manifest_rejects_incompatible_study_configuration(self):
        args = collector_args()
        args.limit = 100
        args.resume = True
        args.overwrite = False
        args.out_root = ""
        manifest = {
            "manifest_version": 5,
            "study_name": STUDY_NAME,
            "platform": "reddit",
            "api_base": API_BASE,
            "target_subreddits": TARGET_SUBREDDITS,
            "age_assurance_queries": AGE_ASSURANCE_QUERIES,
            "relevance_screening": RELEVANCE_SCREENING,
            "event_windows": [{"event_id": event_id, **event} for event_id, event in EVENT_WINDOWS.items()],
            "files": {
                "posts": "posts_raw.jsonl",
                "comments": "comments_raw.jsonl",
                "acquisition_routes": "acquisition_routes.jsonl",
            },
            "args": {"limit": 100, "min_window_seconds": 2, "comment_tree_limit": COMMENT_TREE_LIMIT},
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "collection_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (out / "posts_raw.jsonl").touch()
            (out / "comments_raw.jsonl").touch()
            (out / "acquisition_routes.jsonl").touch()
            validate_resume_manifest(out, args)
            manifest.pop("relevance_screening")
            (out / "collection_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(SystemExit) as screening:
                validate_resume_manifest(out, args)
            self.assertIn("relevance_screening", str(screening.exception))
            manifest["relevance_screening"] = RELEVANCE_SCREENING
            (out / "collection_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (out / "posts_raw.jsonl").unlink()
            with self.assertRaises(SystemExit) as missing:
                validate_resume_manifest(out, args)
            self.assertIn("missing posts_raw.jsonl", str(missing.exception))
            (out / "posts_raw.jsonl").touch()
            manifest["target_subreddits"] = ["privacy"]
            (out / "collection_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(SystemExit) as raised:
                validate_resume_manifest(out, args)
        self.assertIn("target_subreddits", str(raised.exception))

    def test_resume_stats_and_task_history_are_cumulative(self):
        restored = Stats.from_manifest({"requests": 4, "run_requests": 4, "rows_written": 9, "acquisition_matches": 2})
        self.assertEqual(restored.requests, 4)
        self.assertEqual(restored.run_requests, 0)
        self.assertEqual(restored.rows_written, 9)
        self.assertEqual(restored.acquisition_matches, 2)
        previous = [
            {
                "thing": "post_search",
                "subreddit": "privacy",
                "event_id": "australia_legislation",
                "window_start_utc": EVENT_WINDOWS["australia_legislation"]["start_utc"],
                "window_end_utc": EVENT_WINDOWS["australia_legislation"]["end_utc"],
                "query": "age verification",
                "post_id": None,
                "rows_written": 4,
                "requests_used": 2,
                "attempts": 1,
                "status": "partial",
                "error": "request budget reached",
            }
        ]
        current = [
            TaskResult(
                thing="post_search",
                subreddit="privacy",
                event_id="australia_legislation",
                jurisdiction="australia",
                window_start_utc=EVENT_WINDOWS["australia_legislation"]["start_utc"],
                window_end_utc=EVENT_WINDOWS["australia_legislation"]["end_utc"],
                query="age verification",
                status="complete",
                rows_written=3,
                requests_used=1,
            )
        ]
        merged = merge_task_history(previous, current)
        self.assertEqual(merged[0]["rows_written"], 7)
        self.assertEqual(merged[0]["requests_used"], 3)
        self.assertEqual(merged[0]["attempts"], 2)
        self.assertEqual([item["status"] for item in merged[0]["attempt_history"]], ["partial", "complete"])
        self.assertEqual(merged[0]["attempt_history"][0]["error"], "request budget reached")
        self.assertEqual(merged[0]["attempt_history"][1]["requests_used"], 1)

    def test_resume_skips_tasks_already_marked_complete(self):
        planned = event_task()
        target = {
            "post_id": "t3_post",
            "subreddit": planned.subreddit,
            "event_id": planned.event_id,
            "jurisdiction": planned.jurisdiction,
            "window_start_utc": planned.window_start_utc,
            "window_end_utc": planned.window_end_utc,
            "queries": [planned.query, "digital ID"],
        }
        existing_match = {
            "event_id": planned.event_id,
            "jurisdiction": planned.jurisdiction,
            "window_start_utc": planned.window_start_utc,
            "window_end_utc": planned.window_end_utc,
            "query": planned.query,
        }
        previous_tasks = [
            vars(
                TaskResult(
                    thing="post_search",
                    subreddit=planned.subreddit,
                    event_id=planned.event_id,
                    jurisdiction=planned.jurisdiction,
                    window_start_utc=planned.window_start_utc,
                    window_end_utc=planned.window_end_utc,
                    query=planned.query,
                    status="complete",
                )
            ),
            vars(
                TaskResult(
                    thing="comment_tree",
                    subreddit=planned.subreddit,
                    event_id=planned.event_id,
                    jurisdiction=planned.jurisdiction,
                    window_start_utc=planned.window_start_utc,
                    window_end_utc=planned.window_end_utc,
                    post_id=target["post_id"],
                    matched_queries=[planned.query],
                    status="complete",
                )
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "posts_raw.jsonl").touch()
            (out / "comments_raw.jsonl").write_text(
                json.dumps(
                    {
                        "collection": {
                            "thing": "comment",
                            "subreddit": planned.subreddit,
                            "acquisition_matches": [existing_match],
                        },
                        "data": {
                            "id": "t1_comment",
                            "link_id": target["post_id"],
                            "parent_id": target["post_id"],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (out / "acquisition_routes.jsonl").write_text(
                json.dumps(
                    {
                        "thing": "comment",
                        "record_id": "t1_comment",
                        "subreddit": planned.subreddit,
                        "endpoint": "/api/comments/tree",
                        "match": existing_match,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = collector_args(out_root=str(out), resume=True, overwrite=False)
            previous_manifest = {"stats": {"acquisition_matches": 1}, "tasks": previous_tasks, "resume": {"attempt": 1}}
            with patch("src.collect_arctic_shift_core.validate_resume_manifest", return_value=previous_manifest):
                with patch("src.collect_arctic_shift_core.build_plan_summary", return_value=([planned], 1)):
                    with patch("src.collect_arctic_shift_core.load_matched_post_targets", return_value=[target]):
                        with patch(
                            "src.collect_arctic_shift_core.collect_search",
                            return_value=TaskResult(**previous_tasks[0]),
                        ) as collect_search_mock:
                            with patch(
                                "src.collect_arctic_shift_core.collect_comment_tree",
                                return_value=TaskResult(**previous_tasks[1]),
                            ) as collect_tree_mock:
                                _, tasks, _, completion_status, exit_code = run_collection(args)
            manifest = json.loads((out / "collection_manifest.json").read_text(encoding="utf-8"))
            comment = json.loads((out / "comments_raw.jsonl").read_text(encoding="utf-8"))
            routes = [
                json.loads(line)
                for line in (out / "acquisition_routes.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        collect_search_mock.assert_not_called()
        collect_tree_mock.assert_not_called()
        self.assertEqual(tasks, [])
        self.assertEqual(completion_status, "complete")
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(manifest["tasks"]), 2)
        self.assertTrue(all(task["attempts"] == 1 for task in manifest["tasks"]))
        self.assertEqual(set(comment["collection"]["matched_queries"]), {planned.query, "digital ID"})
        self.assertEqual({route["match"]["query"] for route in routes}, {planned.query, "digital ID"})
        comment_task = next(task for task in manifest["tasks"] if task["thing"] == "comment_tree")
        self.assertEqual(comment_task["matched_queries"], [planned.query, "digital ID"])
        self.assertEqual(manifest["stats"]["acquisition_matches"], 2)
        self.assertEqual(manifest["stats"]["comments_written"], 1)
        self.assertEqual(manifest["stats"]["rows_written"], 1)
        self.assertEqual(manifest["stats"]["rows_seen"], 1)

    def test_resume_refreshes_completed_tree_provenance_in_one_pass(self):
        planned = event_task()
        targets = [
            {
                "post_id": f"t3_post_{index}",
                "subreddit": planned.subreddit,
                "event_id": planned.event_id,
                "jurisdiction": planned.jurisdiction,
                "window_start_utc": planned.window_start_utc,
                "window_end_utc": planned.window_end_utc,
                "queries": [planned.query],
            }
            for index in range(2)
        ]
        completed_task_ids = {
            task_identity({"thing": "comment_tree", **target}) for target in targets
        }

        with patch("src.collect_arctic_shift_core.load_matched_post_targets", return_value=targets):
            with patch("src.collect_arctic_shift_core.refresh_comment_tree_provenance") as refresh:
                execute_planned_tasks(
                    planned_tasks=[],
                    tasks=[],
                    posts_file=Path("posts.jsonl"),
                    comments_file=Path("comments.jsonl"),
                    acquisition_file=Path("routes.jsonl"),
                    posts_seen=set(),
                    comments_seen=set(),
                    routes_seen=set(),
                    stats=Stats(),
                    args=collector_args(),
                    completed_task_ids=completed_task_ids,
                )

        refresh.assert_called_once()
        self.assertEqual(refresh.call_args.kwargs["targets"], targets)

    def test_completion_requires_every_search_and_tree_task(self):
        self.assertEqual(compute_completion_status([], expected_count=1), "partial")
        complete = TaskResult(
            thing="post_search",
            subreddit="privacy",
            event_id="australia_legislation",
            jurisdiction="australia",
            window_start_utc="",
            window_end_utc="",
            status="complete",
        )
        self.assertEqual(compute_completion_status([complete], expected_count=2), "partial")
        self.assertEqual(compute_completion_status([complete], expected_count=1), "complete")
        complete.status = "partial"
        self.assertEqual(compute_completion_status([complete], expected_count=1), "partial")
        complete.status = "failed"
        self.assertEqual(compute_completion_status([complete], expected_count=1), "failed")
        self.assertEqual(expected_task_count(270, 4), 274)

    def test_manifest_task_counts_complete_run_has_no_incomplete_tasks(self):
        complete = event_task()
        complete_result = TaskResult(
            thing="post_search",
            subreddit=complete.subreddit,
            event_id=complete.event_id,
            jurisdiction=complete.jurisdiction,
            window_start_utc=complete.window_start_utc,
            window_end_utc=complete.window_end_utc,
            query=complete.query,
            status="complete",
        )
        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(
                Path(tmp),
                collector_args(),
                Stats(),
                [complete_result],
                expected_count=1,
                matched_post_count=0,
                static_task_count=1,
            )
            manifest = json.loads((Path(tmp) / "collection_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["task_counts"],
            {"expected": 1, "recorded": 1, "complete": 1, "partial": 0, "failed": 0, "other": 0, "missing": 0, "incomplete": 0},
        )

    def test_manifest_task_counts_counts_partial_failed_and_missing_tasks(self):
        tasks = []
        for status in ("complete", "partial", "failed", "pending"):
            planned = event_task(status)
            tasks.append(
                TaskResult(
                    thing="post_search",
                    subreddit=planned.subreddit,
                    event_id=planned.event_id,
                    jurisdiction=planned.jurisdiction,
                    window_start_utc=planned.window_start_utc,
                    window_end_utc=planned.window_end_utc,
                    query=planned.query,
                    status=status,
                )
            )
        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(
                Path(tmp),
                collector_args(),
                Stats(),
                tasks,
                expected_count=6,
                matched_post_count=0,
                static_task_count=6,
            )
            manifest = json.loads((Path(tmp) / "collection_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["task_counts"],
            {"expected": 6, "recorded": 4, "complete": 1, "partial": 1, "failed": 1, "other": 1, "missing": 2, "incomplete": 5},
        )

    def test_manifest_task_counts_uses_merged_resume_history(self):
        complete = event_task("complete-history")
        retried = event_task("retry-history")
        failed = event_task("failed-history")

        def result(planned, status):
            return TaskResult(
                thing="post_search",
                subreddit=planned.subreddit,
                event_id=planned.event_id,
                jurisdiction=planned.jurisdiction,
                window_start_utc=planned.window_start_utc,
                window_end_utc=planned.window_end_utc,
                query=planned.query,
                status=status,
            )

        previous_tasks = [vars(result(complete, "complete")), vars(result(retried, "partial")), vars(result(failed, "failed"))]
        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(
                Path(tmp),
                collector_args(),
                Stats(),
                [result(retried, "complete")],
                expected_count=4,
                matched_post_count=0,
                static_task_count=4,
                previous_manifest={"tasks": previous_tasks},
            )
            manifest = json.loads((Path(tmp) / "collection_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["task_counts"],
            {"expected": 4, "recorded": 3, "complete": 2, "partial": 0, "failed": 1, "other": 0, "missing": 1, "incomplete": 2},
        )


if __name__ == "__main__":
    unittest.main()
