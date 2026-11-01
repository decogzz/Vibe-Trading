"""Scheduled manifest-driven paper trading (scheduler unification).

Thin domain surface consumed by the scheduler, mirroring
:mod:`src.br_data.scheduling`: it adds NO trading semantics of its own —
every behavior lives in :mod:`src.tools.paper_manifest_run` and this module
only decides *whether* and *when* the protected system job runs:

* :func:`paper_manifest_job_enabled` is the ``enabled_predicate`` of the
  ``system:paper-manifest-run`` spec in :mod:`src.scheduling.system_jobs`.
  It reads ``VIBE_PAPER_MANIFEST_ENABLED`` (``DataConfig.vibe_paper_manifest_enabled``)
  through :func:`src.config.accessor.get_env_config` — the sanctioned config
  path. The flag defaults to ``False`` and belongs to no preset, so enabling
  the scheduler never silently seeds a job that places paper orders.
* :func:`daily_cron_from_env` is the spec's ``schedule_fn``: it derives the
  5-field cron from ``VIBE_PAPER_MANIFEST_TIME`` ("HH:MM" UTC; default
  ``"21:45"`` → ``"45 21 * * *"``), evaluated lazily inside
  ``ensure_system_jobs`` (at lifespan startup, after dotenv load) — never at
  import time. It NEVER raises: an unparseable value falls back to
  :data:`DEFAULT_DAILY_CRON` with a warning, so one typo'd env var can never
  take the API server down.

*Which manifests the pass trades* is decided per-fire by the runnable itself
from ``VIBE_PAPER_MANIFEST_IDS`` (comma-separated), NOT at seed time —
adding a manifest id to the env takes effect at the next fire without
re-seeding, and an empty value makes each pass a no-op (recorded, not an
error: the executor's retry budget is for failures, not for an empty
watchlist).

Seed-time-only contract: ``ensure_system_jobs`` seeds a row once and a VALID
edited schedule/timezone on that row is trusted afterwards (the system_jobs
local-trust boundary), so ``VIBE_PAPER_MANIFEST_TIME`` drives the FIRST seed
only. The row is a protected system job — REST/CLI PATCH and DELETE refuse
it — so retiming an existing seed means editing the ``schedule`` field of
the stored row in the store JSON file, or deleting the row and restarting.

The runnable itself is :func:`src.tools.paper_manifest_run.run_manifest_paper_pass`
per manifest id, registered under the literal key ``"paper_trade.manifest_run"``
in :mod:`src.scheduling.registry`. It is SYNC by design: the executor's
internal-run supervisor offloads sync callables via ``asyncio.to_thread``,
so the (network-heavy) broker/loader pass never runs on the event loop. The
registry wrapper converts per-manifest failures into an exception naming the
failed manifest ids, so the executor's existing failure path
(``failure_kind="dispatch"``, bounded backoff) records them.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: Fallback cron when ``VIBE_PAPER_MANIFEST_TIME`` is missing or invalid —
#: the same slot the env field's own default ("21:45") derives to.
DEFAULT_DAILY_CRON = "45 */4 * * *"


def paper_manifest_job_enabled() -> bool:
    """Whether the manifest paper-trading system job is enabled.

    Reads ``VIBE_PAPER_MANIFEST_ENABLED`` via the cached ``EnvConfig``
    singleton (default ``False``; no preset raises it). Bound as the
    ``enabled_predicate`` of the ``system:paper-manifest-run`` spec.
    """
    from src.config.accessor import get_env_config

    return get_env_config().data.vibe_paper_manifest_enabled


def manifest_ids_from_env() -> list[str]:
    """Parse ``VIBE_PAPER_MANIFEST_IDS`` into manifest ids (order preserved).

    Read per-fire by the registry runnable — not at seed time — so adding an
    id takes effect at the next fire. Empty/whitespace entries are dropped;
    an empty result makes the pass a recorded no-op.
    """
    from src.config.accessor import get_env_config

    raw = get_env_config().data.vibe_paper_manifest_ids or ""
    return [part.strip() for part in raw.split(",") if part.strip()]


def _every_hours_from_env() -> int:
    """Parse ``VIBE_PAPER_MANIFEST_EVERY_HOURS`` with a safe fallback.

    Returns:
        Hours between pass fires, clamped to 1..24. Garbage falls back to
        the default 4 (crypto's 4H clock) with a warning — the derivation
        must never raise (it runs inside ``ensure_system_jobs``).
    """
    from src.config.accessor import get_env_config

    raw = getattr(get_env_config().data, "vibe_paper_manifest_every_hours", 4)
    try:
        hours = int(raw)
        if not 1 <= hours <= 24:
            raise ValueError(f"out of range: {raw!r}")
    except (TypeError, ValueError):
        logger.warning(
            "paper-manifest: invalid VIBE_PAPER_MANIFEST_EVERY_HOURS %r — "
            "falling back to 4",
            raw,
        )
        return 4
    return hours


def daily_cron_from_env() -> str:
    """Derive the paper-pass cron from the env, never raising.

    Default cadence is EVERY 4 HOURS (crypto trades 24/7 on a 4H bar clock;
    equity orders stay ``day`` and are guarded by the pass's open-order
    check). ``VIBE_PAPER_MANIFEST_EVERY_HOURS=24`` restores once-daily.

    ``VIBE_PAPER_MANIFEST_TIME`` ("HH:MM" UTC) anchors the phase: the cron
    fires at that minute every N hours (``45 */4 * * *`` for the default
    21:45 anchor — the anchor hour itself matters only when N does not
    divide 24).

    Evaluated lazily at ``ensure_system_jobs`` time (the spec's
    ``schedule_fn``); garbage falls back to :data:`DEFAULT_DAILY_CRON`.

    Returns:
        A 5-field cron string, e.g. ``"45 */4 * * *"``.
    """
    from src.config.accessor import get_env_config

    raw = get_env_config().data.vibe_paper_manifest_time
    minute = 45
    hour = 21
    try:
        hh, mm = str(raw).split(":")
        h, m = int(hh), int(mm)
        if 0 <= h <= 23 and 0 <= m <= 59:
            minute, hour = m, h
        else:
            raise ValueError(f"out of range: {raw!r}")
    except (AttributeError, TypeError, ValueError):
        logger.warning(
            "paper-manifest: invalid VIBE_PAPER_MANIFEST_TIME %r — falling back to %s",
            raw,
            DEFAULT_DAILY_CRON,
        )
        return DEFAULT_DAILY_CRON
    every = _every_hours_from_env()
    if every == 24:
        return f"{minute} {hour} * * *"
    if 24 % every == 0:
        # N divides the day: fire at :minute every N hours, phase-anchored
        # by the anchor hour (e.g. 21:45/4 → 45 1,5,9,13,17,21 * * *).
        hours = sorted({(hour + k * every) % 24 for k in range(24 // every)})
        return f"{minute} {','.join(str(h) for h in hours)} * * *"
    # N does not divide 24 (e.g. 5): */5 hours at :minute is the honest form.
    return f"{minute} */{every} * * *"


__all__ = [
    "DEFAULT_DAILY_CRON",
    "daily_cron_from_env",
    "manifest_ids_from_env",
    "paper_manifest_job_enabled",
]