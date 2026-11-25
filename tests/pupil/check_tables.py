"""Before importing the ported schemas: does every class map to an existing table?

Static: parses the class names and tiers from the module source (no adamacs import), maps them to
DataJoint table names, and looks them up in information_schema. Exit code 1 if any is missing.
Run: python tests/pupil/check_tables.py   (read-only; see readonly_db.py)
"""
import re
import sys
from pathlib import Path

import datajoint as dj

sys.path.insert(0, str(Path(__file__).parent))
import readonly_db  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "adamacs" / "schemas"
TIER = {"Lookup": "#", "Manual": "", "Imported": "_", "Computed": "__"}
PRE = "roselab_"
MODULES = {"pupil_tracking.py": {"schema": PRE + "pupil_tracking"},
           "pupil_glint.py": {"schema": PRE + "pupil_tracking"},
           "disk.py": {"schema": PRE + "disk", "pupil_schema": PRE + "pupil_tracking"}}


def tables(path, decorators):
    out, master, deco = [], None, None
    for line in path.read_text().splitlines():
        m = re.match(r"^@(\w+)\s*$", line)
        if m:
            deco = m.group(1)
            continue
        m = re.match(r"^class (\w+)\(dj\.(\w+)\)", line)
        if m:
            if deco in decorators:
                name = TIER[m.group(2)] + dj.utils.from_camel_case(m.group(1))
                master = (decorators[deco], name, m.group(1))
                out.append(master)
            else:
                master = None
            deco = None
            continue
        m = re.match(r"^    class (\w+)\(dj\.Part\)", line)
        if m and master:
            out.append((master[0], master[1] + "__" + dj.utils.from_camel_case(m.group(1)), master[2] + "." + m.group(1)))
    return out


if __name__ == "__main__":
    c = readonly_db.install()
    bad = 0
    for f, decos in MODULES.items():
        for sch, tbl, cls in tables(ROOT / f, decos):
            n = c.query("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema=%s AND table_name=%s",
                        (sch, tbl)).fetchone()[0]
            rows = c.query(f"SELECT COUNT(*) FROM `{sch}`.`{tbl}`").fetchone()[0] if n else None
            print(f"{f}\t{cls}\t{sch}.{tbl}\t{('exists' if n else 'MISSING')}\t{rows}")
            bad += not n
    sys.exit(1 if bad else 0)
