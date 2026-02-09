"""Notebook runtime helpers for reliable DataJoint bootstrap in ingest notebooks."""

from __future__ import annotations

import os
import shutil
import sys
import logging
from dataclasses import dataclass
from pathlib import Path

import datajoint as dj


_TLS_HANDSHAKE_MARKERS = (
    "sslv3_alert_handshake_failure",
    "ssl/tls alert handshake failure",
    "tlsv1 alert handshake failure",
    "ssl handshake failure",
)

_DATAJOINT_LOGGER_NAMES = (
    "datajoint",
    "datajoint.connection",
    "datajoint.settings",
    "datajoint.plugin",
)


@dataclass(frozen=True)
class IngestNotebookContext:
    """Resolved runtime context for ingest notebooks."""

    repo_root: Path
    config_path: Path
    tls_fallback_applied: bool
    dot_path: str | None


def _looks_like_tls_handshake_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return ("ssl" in text and "handshake" in text) or any(
        marker in text for marker in _TLS_HANDSHAKE_MARKERS
    )


def _resolve_repo_root(cwd: Path) -> Path:
    candidates = [cwd, *cwd.parents]
    return next(
        (
            p
            for p in candidates
            if (p / "setup.py").exists() and (p / "adamacs").exists()
        ),
        cwd,
    )


def _load_config(repo_root: Path) -> Path:
    candidates = [
        repo_root / "dj_local_conf.json",
    ]
    for path in candidates:
        if path.exists():
            dj.config.load(str(path))
            return path
    raise FileNotFoundError("Could not find dj_local_conf.json in ingest repository root.")


def _set_datajoint_log_level(level: int = logging.WARNING) -> None:
    for logger_name in _DATAJOINT_LOGGER_NAMES:
        logging.getLogger(logger_name).setLevel(level)


def _synchronize_package_db_prefix() -> None:
    """
    Ensure adamacs.db_prefix reflects the loaded dj.config custom prefix.

    `adamacs` is imported before config loading when importing `adamacs.notebook_runtime`,
    so db_prefix may otherwise stay on the package default.
    """
    package_module = sys.modules.get("adamacs")
    if package_module is None:
        return
    custom_prefix = dj.config.get("custom", {}).get(
        "database.prefix", getattr(package_module, "default_prefix", "adamacs_")
    )
    package_module.db_prefix = custom_prefix


def _connect_with_tls_fallback(*, allow_tls_fallback: bool) -> bool:
    tls_fallback_applied = False
    # Default to non-TLS when the config leaves this unset (null/None).
    if dj.config.get("database.use_tls") is None:
        dj.config["database.use_tls"] = False
    try:
        dj.conn(reset=True)
    except Exception as exc:
        if not allow_tls_fallback or not _looks_like_tls_handshake_error(exc):
            raise
        dj.config["database.use_tls"] = False
        dj.conn(reset=True)
        tls_fallback_applied = True
    return tls_fallback_applied


def bootstrap_ingest_notebook(
    *,
    connect: bool = True,
    allow_tls_fallback: bool = True,
    quiet_datajoint_logs: bool = True,
    verbose: bool = True,
) -> IngestNotebookContext:
    """
    Prepare notebook runtime:
    - resolve ingest repository root
    - load local DataJoint config
    - connect to database with TLS fallback for SSL handshake failures
    """

    cwd = Path.cwd()
    repo_root = _resolve_repo_root(cwd)
    os.chdir(repo_root)

    if quiet_datajoint_logs:
        _set_datajoint_log_level(logging.WARNING)
    config_path = _load_config(repo_root)
    _synchronize_package_db_prefix()
    if quiet_datajoint_logs:
        _set_datajoint_log_level(logging.WARNING)
    tls_fallback_applied = _connect_with_tls_fallback(
        allow_tls_fallback=allow_tls_fallback
    ) if connect else False

    context = IngestNotebookContext(
        repo_root=repo_root,
        config_path=config_path,
        tls_fallback_applied=tls_fallback_applied,
        dot_path=shutil.which("dot"),
    )

    if verbose:
        print(f"Ingest root:   {context.repo_root}")
        print(f"Config path:   {context.config_path}")
        print(f"DataJoint:     {dj.__version__}")
        print(f"TLS fallback:  {context.tls_fallback_applied}")
        print(f"Graphviz dot:  {context.dot_path or 'NOT FOUND'}")

    return context
