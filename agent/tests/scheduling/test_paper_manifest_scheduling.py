"""Regression tests: scheduled manifest-driven paper trading (system job wiring).

Scheduler-unification contract (mirrors tests/scheduling/test_system_jobs.py
and src/br_data/scheduling.py's tests):

- Spec surface: the system:paper-manifest-run spec is well-formed, its
  registry key resolves to the runnable, and nothing seeds while
  VIBE_PAPER_MANIFEST_ENABLED is off (default).
- Cron derivation: lazy, never raises, in-range fallback for garbage.
- Manifest ids: parsed per-fire from VIBE_PAPER_MANIFEST_IDS; empty = no-op.
- The runnable: one run_manifest_paper_pass per id, per-manifest failure
  aggregation into one dispatch exception, empty list = recorded no-op.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scheduling import paper_manifest_scheduling as pms
from src.scheduling import registry
from src.scheduling.registry import resolve_internal
from src.scheduling.system_jobs import (
    SYSTEM_JOB_ID_PREFIX,
    SystemJobSpec,
    ensure_system_jobs,
    is_system_job_id,
    system_job_specs,
)
from src.scheduled_research.store import ScheduledResearchJobStore


# ---------------------------------------------------------------------------
# Spec surface
# ---------------------------------------------------------------------------

def test_paper_manifest_spec_exists_and_is_wellformed() -> None:
    specs = {s.id: s for s in system_job_specs()}
    spec = specs.get("system:paper-manifest-run")
    assert spec is not None, "paper-manifest spec missing from _SYSTEM_JOB_SPECS"
    assert spec.internal_callable == "paper_trade.manifest_run"
    assert spec.executor_kind == "internal"
    assert spec.source_type == "system"
    assert spec.prompt == ""
    assert callable(spec.enabled_predicate)
    assert callable(spec.schedule_fn)


def test_paper_manifest_registry_key_resolves() -> None:
    fn = resolve_internal("paper_trade.manifest_run")
    assert callable(fn)
    assert "paper_trade.manifest_run" in registry.list_internal()


def test_spec_id_is_unforgeable_prefix_shape() -> None:
    assert is_system_job_id("system:paper-manifest-run") is True
    assert ":" in SYSTEM_JOB_ID_PREFIX  # SAFE_JOB_ID_RE anti-forgery interplay


def test_disabled_default_seeds_nothing(tmp_path: Path) -> None:
    """The flag defaults to False and belongs to no preset: enabling the
    scheduler never silently seeds a job that places paper orders."""
    assert pms.paper_manifest_job_enabled() is False
    store = ScheduledResearchJobStore(path=tmp_path / "jobs.json")
    report = ensure_system_jobs(store)
    assert "system:paper-manifest-run" in report.skipped_disabled
    assert "system:paper-manifest-run" not in report.seeded
    assert not (tmp_path / "jobs.json").exists()


def test_enabled_spec_seeds_the_row(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pms, "paper_manifest_job_enabled", lambda: True)
    # system_jobs imported the predicate by reference at module import —
    # patch where it consumed it
    import src.scheduling.system_jobs as sj

    monkeypatch.setattr(
        "src.scheduling.paper_manifest_scheduling.paper_manifest_job_enabled",
        lambda: True,
    )
    # rebuild specs with the patched predicate
    spec = SystemJobSpec(
        id="system:paper-manifest-run",
        schedule_fn=pms.daily_cron_from_env,
        timezone="UTC",
        internal_callable="paper_trade.manifest_run",
        title="Manifest paper trading",
        enabled_predicate=lambda: True,
    )
    monkeypatch.setattr(sj, "_SYSTEM_JOB_SPECS", (spec,))
    store = ScheduledResearchJobStore(path=tmp_path / "jobs.json")
    report = ensure_system_jobs(store)
    assert report.seeded == ("system:paper-manifest-run",)
    job = store.get("system:paper-manifest-run")
    assert job is not None
    assert job.internal_callable == "paper_trade.manifest_run"
    # cron with timezone waits for the next slot, not "now"
    assert job.next_run_at > 0


# ---------------------------------------------------------------------------
# Cron derivation (lazy schedule_fn)
# ---------------------------------------------------------------------------

def test_daily_cron_default_slot() -> None:
    assert pms.daily_cron_from_env() == "45 21 * * *"


@pytest.mark.parametrize("raw,expected", [
    ("09:30", "30 9 * * *"),
    ("00:00", "0 0 * * *"),
    ("23:59", "59 23 * * *"),
])
def test_daily_cron_parses_valid_times(monkeypatch, raw, expected) -> None:
    class _Cfg:
        class data:  # noqa: N805
            vibe_paper_manifest_time = raw
    monkeypatch.setattr(
        "src.config.accessor.get_env_config", lambda: _Cfg()
    )
    assert pms.daily_cron_from_env() == expected


@pytest.mark.parametrize("bad", ["25:99", "", "abc", "7", "1:2:3", None])
def test_daily_cron_never_raises_on_garbage(monkeypatch, bad) -> None:
    class _Cfg:
        class data:  # noqa: N805
            vibe_paper_manifest_time = bad
    monkeypatch.setattr(
        "src.config.accessor.get_env_config", lambda: _Cfg()
    )
    assert pms.daily_cron_from_env() == pms.DEFAULT_DAILY_CRON


# ---------------------------------------------------------------------------
# Manifest ids (per-fire parse)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("", []),
    ("   ", []),
    ("abc", ["abc"]),
    ("a, b ,c", ["a", "b", "c"]),
    (" x ,,y ", ["x", "y"]),
])
def test_manifest_ids_parse(monkeypatch, raw, expected) -> None:
    class _Cfg:
        class data:  # noqa: N805
            vibe_paper_manifest_ids = raw
    monkeypatch.setattr(
        "src.config.accessor.get_env_config", lambda: _Cfg()
    )
    assert pms.manifest_ids_from_env() == expected


# ---------------------------------------------------------------------------
# The runnable (paper_trade.manifest_run)
# ---------------------------------------------------------------------------

def test_runnable_empty_ids_is_recorded_noop(monkeypatch, tmp_path) -> None:
    """Empty VIBE_PAPER_MANIFEST_IDS: no pass runs, a no-op note is audited,
    and the callable returns without raising (no retry-budget burn)."""
    monkeypatch.setattr(pms, "manifest_ids_from_env", lambda: [])
    called = []
    monkeypatch.setattr(
        "src.tools.paper_manifest_run.run_manifest_paper_pass",
        lambda mid: called.append(mid),
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    fn = resolve_internal("paper_trade.manifest_run")
    fn()  # must not raise

    assert called == []
    log = tmp_path / ".vibe-trading" / "paper_pass_log.jsonl"
    assert log.exists()
    last = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert "VIBE_PAPER_MANIFEST_IDS empty" in last["note"]


def test_runnable_runs_each_manifest_once(monkeypatch) -> None:
    ids = ["m1", "m2", "m3"]
    monkeypatch.setattr(pms, "manifest_ids_from_env", lambda: ids)
    called = []
    monkeypatch.setattr(
        "src.tools.paper_manifest_run.run_manifest_paper_pass",
        lambda mid: called.append(mid),
    )
    fn = resolve_internal("paper_trade.manifest_run")
    fn()
    assert called == ids


def test_runnable_aggregates_failures_and_raises(monkeypatch) -> None:
    """One drifted manifest must not abort the others; the pass records
    which ids failed and raises ONE exception (dispatch failure path)."""
    monkeypatch.setattr(pms, "manifest_ids_from_env", lambda: ["ok1", "bad", "ok2"])

    def fake_pass(mid: str) -> str:
        if mid == "bad":
            raise RuntimeError("strategy drifted since compilation")
        return json.dumps({"status": "ok", "manifest_id": mid})

    monkeypatch.setattr(
        "src.tools.paper_manifest_run.run_manifest_paper_pass", fake_pass
    )
    fn = resolve_internal("paper_trade.manifest_run")
    with pytest.raises(RuntimeError, match="bad"):
        fn()  # ok1 and ok2 must have run before the raise


def test_runnable_is_sync_callable() -> None:
    import inspect

    fn = resolve_internal("paper_trade.manifest_run")
    assert not inspect.iscoroutinefunction(fn), "executor offloads via to_thread"