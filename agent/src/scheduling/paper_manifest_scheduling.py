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
DEFAULT_DAILY_CRON = "45 21 * * *"


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


def daily_cron_from_env() -> str:
    """Derive the daily cron (``"M H * * *"``) from ``VIBE_PAPER_MANIFEST_TIME``.

    Evaluated lazily at ``ensure_system_jobs`` time (the spec's
    ``schedule_fn``), so it must NEVER raise: any value that cannot parse as
    an in-range "HH:MM" falls back to :data:`DEFAULT_DAILY_CRON` with a
    warning.

    Returns:
        A 5-field cron string, e.g. ``"45 21 * * *"`` for the default
        21:45 UTC slot.
    """
    from src.config.accessor import get_env_config

    raw = get_env_config().data.vibe_paper_manifest_time
    try:
        hh, mm = str(raw).split(":")
        hour, minute = int(hh), int(mm)
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError(f"out of range: {raw!r}")
    except (AttributeError, TypeError, ValueError):
        logger.warning(
            "paper-manifest: invalid VIBE_PAPER_MANIFEST_TIME %r — falling back to %s",
            raw,
            DEFAULT_DAILY_CRON,
        )
        return DEFAULT_DAILY_CRON
    return f"{minute} {hour} * * *"


__all__ = [
    "DEFAULT_DAILY_CRON",
    "daily_cron_from_env",
    "manifest_ids_from_env",
    "paper_manifest_job_enabled",
]