"""Read-only DataJoint access for the pupil / glint / DISK golden tests.

Importing the adamacs pipeline with a normal connection can write: Schema.activate creates
missing schemas and tables, and a Lookup whose `contents` outnumber its rows is filled with
INSERT IGNORE. These tests only ever read, so before anything is imported this module
  * forces every Schema.activate / Schema(cls) to bind existing tables only,
  * lets Table.insert / insert1 pass ONLY when every row already exists (that is what the
    import-time Lab / User / Project / Lookup inserts amount to on the live database) and
    otherwise raises, without sending any SQL,
  * blocks delete, drop, update1, alter, declare and populate,
  * admits only read-only SQL, and puts the session in READ ONLY transaction mode, so the
    server refuses a write even if something slips past the patches.
"""
import json
import os
import re
from pathlib import Path

import datajoint as dj

CONFIG = Path(os.environ.get("ADAMACS_DJ_CONFIG", Path.home() / "adamacs" / "dj_local_conf.json"))
_ALLOWED = re.compile(r"^\s*(SELECT|SHOW|DESCRIBE|DESC|EXPLAIN|SET\s+SESSION\s+TRANSACTION\s+READ\s+ONLY"
                      r"|SET\s+(SESSION\s+)?(sql_mode|time_zone|character_set|names|autocommit)\b|USE\s)", re.I)
_installed = False


class ReadOnlyViolation(RuntimeError):
    pass


def available():
    return CONFIG.exists()


def _blocked(name):
    def f(*a, **k):
        raise ReadOnlyViolation(f"{name} is blocked in the read-only tests")
    return f


def install():
    """Patch DataJoint (idempotent) and connect read-only. Call before importing adamacs."""
    global _installed
    if _installed:
        return dj.conn()
    from datajoint.table import Table
    from datajoint.autopopulate import AutoPopulate
    from datajoint.connection import Connection
    from datajoint.schemas import Schema

    cfg = json.loads(CONFIG.read_text())
    dj.config.update({k: v for k, v in cfg.items() if k.startswith("database.") or k in
                      ("stores", "enable_python_native_blobs", "fetch_format", "filepath_checksum_size_limit")})
    dj.config["custom"] = cfg.get("custom", {"database.prefix": "roselab_"})
    dj.config["safemode"] = True

    def _insert_existing_only(self, rows, *a, **k):
        rows = [rows] if isinstance(rows, dict) else list(rows)
        if rows and not isinstance(rows[0], dict):
            rows = [dict(zip(self.heading.names, r)) for r in rows]
        missing = [r for r in rows if not len(self & {pk: r[pk] for pk in self.primary_key})]
        if missing:
            raise ReadOnlyViolation(f"insert into {self.full_table_name} would add {len(missing)} row(s)")

    def _insert1_existing_only(self, row, *a, **k):
        _insert_existing_only(self, [row])

    Table.insert = _insert_existing_only
    Table.insert1 = _insert1_existing_only
    for n in ("delete", "delete_quick", "drop", "drop_quick", "update1", "alter", "declare"):
        setattr(Table, n, _blocked(f"Table.{n}"))
    AutoPopulate.populate = _blocked("AutoPopulate.populate")

    orig_activate, orig_call = Schema.activate, Schema.__call__

    def activate(self, schema_name=None, *, connection=None, create_schema=None, create_tables=None, add_objects=None):
        return orig_activate(self, schema_name, connection=connection, create_schema=False,
                             create_tables=False, add_objects=add_objects)

    def call(self, cls, *, context=None):
        self.create_tables = False
        return orig_call(self, cls, context=context)

    Schema.activate, Schema.__call__ = activate, call

    orig_query = Connection.query

    def query(self, sql, args=(), *, as_dict=False, suppress_warnings=True, reconnect=None):
        stripped = re.sub(r"^\s*(/\*.*?\*/|--[^\n]*\n)\s*", "", sql, flags=re.S)
        if not _ALLOWED.match(stripped):
            raise ReadOnlyViolation("blocked SQL: " + stripped.split(None, 1)[0])
        return orig_query(self, sql, args, as_dict=as_dict, suppress_warnings=suppress_warnings, reconnect=reconnect)

    Connection.query = query
    c = dj.conn()
    c.query("SET SESSION TRANSACTION READ ONLY")
    _installed = True
    return c


class capture_inserts:
    """Within the block, Table.insert / insert1 record the rows they are given and send nothing.
    Lets a make() run end to end so its output can be compared with the stored row."""

    def __enter__(self):
        from datajoint.table import Table
        self.rows, self._saved = [], (Table.insert, Table.insert1)
        rows = self.rows

        def insert(tbl, r, *a, **k):
            r = [r] if isinstance(r, dict) else list(r)
            rows.extend((tbl.table_name, dict(x)) for x in r)

        def insert1(tbl, r, *a, **k):
            rows.append((tbl.table_name, dict(r)))

        Table.insert, Table.insert1 = insert, insert1
        return self

    def __exit__(self, *exc):
        from datajoint.table import Table
        Table.insert, Table.insert1 = self._saved
        return False

    def of(self, table_name):
        return [r for t, r in self.rows if t == table_name]


def load_pipeline(root):
    """Import the ported adamacs.pipeline from `root` read-only, and bind the DISK schema."""
    import sys
    install()
    root = os.path.realpath(str(root))
    if root not in sys.path:
        sys.path.insert(0, root)
    cwd = os.getcwd()
    os.chdir(root)
    try:
        import adamacs
        assert os.path.realpath(adamacs.__file__).startswith(root), adamacs.__file__
        import adamacs.pipeline as P
    finally:
        os.chdir(cwd)
    return P
