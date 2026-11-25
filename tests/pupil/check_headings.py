"""After check_tables.py passes: import the ported pipeline read-only and compare every class
definition (attributes, types, nullability, primary key) with the live table heading."""
import re
import sys
from pathlib import Path

import datajoint as dj
from datajoint.declare import prepare_declare

sys.path.insert(0, str(Path(__file__).parent))
import readonly_db  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
P = readonly_db.load_pipeline(ROOT)
from adamacs.schemas import disk, pupil_glint, pupil_tracking  # noqa: E402


def classes(mod):
    for n in dir(mod):
        c = getattr(mod, n)
        if isinstance(c, type) and issubclass(c, dj.Table) and c.__module__ == mod.__name__ and hasattr(c, "definition") \
                and getattr(c, "database", None):
            yield c
            for pn in dir(c):
                p = getattr(c, pn)
                if isinstance(p, type) and issubclass(p, dj.Part) and p is not dj.Part:
                    yield p


def norm(t):
    t = t.lower().replace(" ", "")
    return {"int": "int", "int(11)": "int", "intunsigned": "intunsigned", "int(10)unsigned": "intunsigned",
            "smallint(6)": "smallint", "smallint(5)unsigned": "smallintunsigned", "tinyint(4)": "tinyint",
            "tinyint(3)unsigned": "tinyintunsigned", "bigint(20)": "bigint"}.get(t, t)


bad = 0
for mod in (pupil_tracking, pupil_glint, disk):
    for c in classes(mod):
        t = c()
        ctx = dict(vars(mod)); ctx.update(getattr(mod, "_objects", {}) or {})
        if mod is disk:
            ctx.update({k: getattr(disk._linking_module, k) for k in dir(disk._linking_module) if not k.startswith("_")})
            ctx.update(PupilEllipseParameterFreeMoving=disk._param_table, PupilEllipseParameter=disk._headfixed_param_table,
                       **{n: disk.upstream(n) for n in disk._UPSTREAM_DECLARED})
        if issubclass(c, dj.Part):
            ctx[c.__qualname__.split(".")[0]] = getattr(mod, c.__qualname__.split(".")[0])
            ctx["master"] = getattr(mod, c.__qualname__.split(".")[0])
        defn = c.definition if not callable(c.definition) else t.definition
        try:
            _, pk, attr_sql, *_ = prepare_declare(defn, ctx)
        except Exception as e:  # noqa: BLE001
            print(f"PARSE-FAIL\t{c.__qualname__}\t{e}"); bad += 1; continue
        decl = {}
        for s in attr_sql:
            m = re.match(r"`(\w+)`\s+(.+?)\s+(NOT NULL|DEFAULT|NULL)", s)
            typ = m.group(2)
            if typ == "binary(16)" and re.search(r"COMMENT\s+\":(blob|attach|filepath)@", s):   # external store: uuid column
                typ = re.search(r"COMMENT\s+\":([\w@-]+):", s).group(1)
            decl[m.group(1)] = (norm(typ), m.group(3) != "NOT NULL")
        live = {a.name: (norm(a.sql_type if hasattr(a, "sql_type") else a.type), a.nullable) for a in t.heading.attributes.values()}
        diffs = []
        if set(pk) != set(t.primary_key):
            diffs.append(f"pk {pk} vs {t.primary_key}")
        for n in sorted(set(decl) | set(live)):
            if n not in live:
                diffs.append(f"{n}: not in table")
            elif n not in decl:
                diffs.append(f"{n}: only in table")
            elif decl[n] != live[n]:
                diffs.append(f"{n}: {decl[n]} vs {live[n]}")
        print(("OK" if not diffs else "DIFF") + "\t" + mod.__name__.split(".")[-1] + f"\t{c.__qualname__}\t{len(live)} attrs\t" + "; ".join(diffs))
        bad += bool(diffs)
        if isinstance(t, dj.Lookup) and hasattr(c, "contents"):
            print(f"\tcontents {len(list(c.contents))} <= rows {len(t)}: {len(list(c.contents)) <= len(t)}")
print("headings differing:", bad)
sys.exit(1 if bad else 0)
