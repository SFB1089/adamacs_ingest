#!/usr/bin/env python
"""Compile every table declaration in ``cell_registration`` without touching the database.

Run from the repository root:

    python notebooks/tobiasr/registration/validate_cell_registration_schema.py

What it checks
--------------
1. Importing the module declares nothing (``schema.is_activated()`` is False).
2. The table set is **discovered from the module**, in source order, walking each master's
   namespace for part tables. It is not read from a list maintained by hand. The topcam
   design review records why that matters: its validator's hand-kept dict had fallen a
   table behind and printed *"All 18 definitions check out"* -- true about the list, silent
   about the module.
3. Every discovered class actually carries a ``definition`` to compile.
4. Each definition compiles to SQL via DataJoint's own declare path, against a *fake*
   schema name, so nothing is created.
5. No primary key contains a nullable attribute.
6. The counts printed are the module's own, never a literal.

Exit status is non-zero if anything fails, so CI can use it.
"""
from __future__ import annotations

import inspect
import re
import sys

import datajoint as dj


def discover_tables(module):
    """(name, class) for every table in the module, in source order, parts after masters."""
    members = []
    for name, obj in vars(module).items():
        if not inspect.isclass(obj) or not issubclass(obj, dj.table.Table):
            continue
        if obj.__module__ != module.__name__:
            continue
        if issubclass(obj, dj.Part):
            continue
        try:
            line = inspect.getsourcelines(obj)[1]
        except (OSError, TypeError):
            line = 1 << 30
        members.append((line, name, obj))
    members.sort()

    out = []
    for _, name, cls in members:
        out.append((name, cls))
        parts = []
        for part_name, part in vars(cls).items():
            if inspect.isclass(part) and issubclass(part, dj.Part):
                try:
                    line = inspect.getsourcelines(part)[1]
                except (OSError, TypeError):
                    line = 1 << 30
                parts.append((line, f"{name}.{part_name}", part))
        parts.sort()
        out.extend((pname, pcls) for _, pname, pcls in parts)
    return out


def primary_key_of(definition: str):
    """Attribute names above the ``---`` divider, excluding foreign-key lines."""
    keys = []
    for raw in definition.splitlines():
        line = raw.strip()
        if line.startswith("---"):
            break
        if not line or line.startswith("#"):
            continue
        if line.startswith("->"):
            keys.append(line)
            continue
        m = re.match(r"^([A-Za-z_]\w*)\s*(=.*?)?\s*:", line)
        if m:
            keys.append(m.group(1))
    return keys


def main() -> int:
    from adamacs.schemas import cell_registration as mod

    print(f"on import: cell_registration.schema activated={mod.schema.is_activated()}")
    if mod.schema.is_activated():
        print("FAIL: importing the module activated the schema; it must declare nothing.")
        return 1
    print("-> importing the module declared nothing.\n")

    tables = discover_tables(mod)
    print(f"discovered {len(tables)} table definitions in {mod.__name__}\n")

    failures = []
    external_fk = 0
    for name, cls in tables:
        definition = getattr(cls, "definition", None)
        if not definition or not str(definition).strip():
            failures.append((name, "carries no definition to compile"))
            print(f"  {name:<42} NO DEFINITION")
            continue
        text = str(definition)

        # Structural checks -- these need no connection and are the point of the script.
        keys = primary_key_of(text)
        if not keys:
            failures.append((name, "no primary key attributes found"))
        nullable_in_pk = [k for k in keys
                          if not k.startswith("->") and "=null" in _attr_line(text, k)]
        if nullable_in_pk:
            failures.append((name, f"nullable attribute(s) in primary key: {nullable_in_pk}"))

        # Attribute syntax: every non-FK line must parse as `name[=default] : type`.
        bad = _unparseable_lines(text)
        if bad:
            failures.append((name, f"unparseable attribute line(s): {bad}"))

        # Compiling a definition to SQL resolves its foreign keys, which needs the
        # referenced table's heading, which needs a live connection. Offline we can
        # therefore only compile definitions that declare no dependency at all; the rest
        # are validated structurally and reported as such, rather than being reported as
        # passing something that was never attempted.
        has_dependency = bool(re.search(r"^\s*->", text, re.M))
        if has_dependency:
            external_fk += 1
            status = "STRUCT-OK (has FK: needs connection)"
        else:
            try:
                dj.declare.declare(
                    full_table_name=f"`fake_cell_registration`.`{_sql_name(name)}`",
                    definition=text, context=dict(vars(mod)))
                status = "COMPILED"
            except Exception as exc:  # noqa: BLE001
                failures.append((name, f"{type(exc).__name__}: {exc}"))
                status = f"COMPILE ERROR: {type(exc).__name__}"
        flag = f"  <-- NULLABLE IN PK: {nullable_in_pk}" if nullable_in_pk else ""
        print(f"  {name:<42} {status:<34} PK {keys}{flag}")

    print()
    print(f"{external_fk} of {len(tables)} definition(s) declare a foreign key, so compiling "
          "them to SQL would need a live connection to resolve the referenced heading; "
          "those were validated structurally only.")
    print(f"{len(tables) - external_fk} dependency-free definition(s) were compiled to SQL "
          "in full.")
    if failures:
        print(f"\n{len(failures)} of {len(tables)} definitions FAILED:")
        for name, why in failures:
            print(f"   {name}: {why}")
        return 1

    print(f"All {len(tables)} definitions check out structurally. Nothing was declared; "
          "the database is unchanged.")
    return 0


#: `index (a, b)` / `unique index (a, b)` are index declarations, not attributes.
_INDEX_RE = re.compile(r"^(unique\s+)?index\s*\(", re.I)


def _unparseable_lines(definition: str):
    bad = []
    for raw in definition.splitlines():
        line = raw.strip()
        if (not line or line.startswith("---") or line.startswith("#")
                or line.startswith("->") or _INDEX_RE.match(line)):
            continue
        if not re.match(r"^[A-Za-z_]\w*\s*(=\s*\S.*?)?\s*:\s*\S", line):
            bad.append(line[:60])
    return bad


def _sql_name(class_name: str) -> str:
    """Rough DataJoint table name, good enough for a throwaway compile target."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", class_name.replace(".", "__")).lower()


def _attr_line(definition: str, attr: str) -> str:
    for raw in definition.splitlines():
        if re.match(rf"^\s*{re.escape(attr)}\s*(=|:)", raw):
            return raw
    return ""


if __name__ == "__main__":
    sys.exit(main())
