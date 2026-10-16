"""Draw the cell_registration schema and workflow diagrams.

The table set, tiers, keys and foreign keys are **parsed out of
``adamacs/schemas/cell_registration.py`` itself**, never listed here by hand. A diagram
that is maintained separately from the schema drifts from it silently; the topcam design
review records that failure mode for a validator whose hand-kept list had fallen a table
behind and still reported success.

Needs graphviz (``conda install -c conda-forge graphviz python-graphviz``). Writes PNG and
DOT beside the output directory given on the command line; ``notebooks/figures/*.png`` is
gitignored, so regenerate rather than expecting the images in the tree.

    python scripts/draw_cell_registration_diagrams.py notebooks/figures
"""
from __future__ import annotations

import os
import re
import sys

import graphviz

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = os.path.join(HERE, os.pardir, "adamacs", "schemas", "cell_registration.py")

FILL = {"dj.Lookup": "#ececec", "dj.Manual": "#d9ead3", "dj.Imported": "#d6e4f0",
        "dj.Computed": "#f4dada", "dj.Part": "#ffffff"}
LINE = {"dj.Lookup": "#8a8a8a", "dj.Manual": "#5e8b52", "dj.Imported": "#436d96",
        "dj.Computed": "#a35757", "dj.Part": "#9a9a9a"}


def read_tables(src: str):
    """[(full_name, tier, master_or_None)] in source order, parts after their master."""
    masters = [(m.group(1), m.group(2), m.start())
               for m in re.finditer(r"^@schema\s*\n\s*class\s+(\w+)\((dj\.\w+)[^)]*\):",
                                    src, re.M)]
    parts = [(m.group(1), "dj.Part", m.start())
             for m in re.finditer(r"^    class\s+(\w+)\(dj\.Part\):", src, re.M)]
    out, current = [], None
    for name, tier, _ in sorted(masters + parts, key=lambda t: t[2]):
        if tier != "dj.Part":
            current = name
            out.append((name, tier, None))
        else:
            out.append((f"{current}.{name}", tier, current))
    return out


def read_definition(src: str, cls: str, is_part: bool) -> str:
    pat = (rf"class\s+{cls}\(dj\.Part\):\s*\n\s*definition\s*=\s*\"\"\"(.*?)\"\"\""
           if is_part else
           rf"^@schema\s*\n\s*class\s+{cls}\(dj\.\w+[^)]*\):\s*\n(?:.*?\n)*?"
           rf"\s*definition\s*=\s*\"\"\"(.*?)\"\"\"")
    m = re.search(pat, src, re.S | re.M)
    return m.group(1) if m else ""


def parse_definition(text: str) -> dict:
    """Split a DataJoint definition into primary-key and secondary parts.

    Comments are stripped FIRST: ``# sha1 over the projected (mask -> value) rows`` would
    otherwise be read as a foreign key to a table called ``value``.
    """
    lines = [ln.split("#", 1)[0].rstrip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln.strip()]
    sep = next((i for i, ln in enumerate(lines) if set(ln.strip()) == {"-"}), len(lines))

    def foreign_keys(block):
        found = []
        for ln in block:
            m = re.match(r"\s*->\s*(\[[^\]]*\]\s*)?([\w.]+)", ln)
            if m:
                found.append((m.group(2), bool(m.group(1) and "nullable" in m.group(1))))
        return found

    def attributes(block):
        return [re.match(r"\s*(\w+)", ln).group(1) for ln in block
                if not ln.strip().startswith("->") and re.match(r"\s*\w+\s*(=[^:]*)?:", ln)]

    pk, sec = lines[:sep], lines[sep + 1:]
    return {"pk_fk": foreign_keys(pk), "sec_fk": foreign_keys(sec),
            "pk_attr": attributes(pk), "sec_attr": attributes(sec)}


def node_id(name: str) -> str:
    return name.replace(".", "__")


def node_label(full: str, parsed: dict, master: str | None) -> str:
    keyed = [n.split(".")[-1] for n, _ in parsed["pk_fk"] if n != "master"]
    bits = []
    if master:
        bits.append(f"<i>part of</i> {master}")
    if keyed:
        bits.append("&#8594; " + ", ".join(keyed))
    if parsed["pk_attr"]:
        shown = ", ".join(parsed["pk_attr"][:3])
        bits.append(shown + ("&#8230;" if len(parsed["pk_attr"]) > 3 else ""))
    sub = "<br/>".join(f'<font point-size="7.5" color="#555555">{b}</font>' for b in bits)
    return f"<<b>{full.split('.')[-1]}</b>{'<br/>' + sub if sub else ''}>"


def build_erd(tables, parsed):
    g = graphviz.Digraph("cell_registration", format="png")
    g.attr(rankdir="TB", bgcolor="white", fontname="Helvetica", dpi="300",
           ranksep="0.40", nodesep="0.26", margin="0.12", splines="spline")
    g.attr("node", shape="box", style="filled,rounded", fontname="Helvetica",
           fontsize="10", penwidth="1.1", margin="0.12,0.07")
    g.attr("edge", fontname="Helvetica", fontsize="8", arrowsize="0.6", color="#5a5a5a")

    external = sorted({dep for p in parsed.values()
                       for dep, _ in p["pk_fk"] + p["sec_fk"] if "." in dep
                       and dep.split(".")[0] not in {t.split(".")[0] for t, _, _ in tables}})
    for dep in external:
        g.node(node_id(dep), f"<<b>{dep}</b>>", fillcolor="#fdf2d8", color="#c89f3c",
               style="filled,rounded,dashed", penwidth="1.4")

    for full, tier, master in tables:
        if master:
            continue
        kids = [f for f, _, m in tables if m == full]
        if kids:
            with g.subgraph(name=f"cluster_{node_id(full)}") as cl:
                cl.attr(style="rounded", color="#cdcdcd", penwidth="0.9",
                        bgcolor="#fbfbfb", margin="8", label="")
                cl.node(node_id(full), node_label(full, parsed[full], None),
                        fillcolor=FILL[tier], color=LINE[tier])
                for k in kids:
                    cl.node(node_id(k), node_label(k, parsed[k], full),
                            fillcolor=FILL["dj.Part"], color=LINE["dj.Part"],
                            fontsize="9", penwidth="0.8")
        else:
            g.node(node_id(full), node_label(full, parsed[full], None),
                   fillcolor=FILL[tier], color=LINE[tier])

    for full, tier, master in tables:
        if master:
            g.edge(node_id(master), node_id(full), arrowhead="none",
                   color="#b0b0b0", penwidth="2.0")
        for dep, _ in parsed[full]["pk_fk"]:
            if dep != "master":
                g.edge(node_id(dep), node_id(full), penwidth="1.1")
        for dep, nullable in parsed[full]["sec_fk"]:
            g.edge(node_id(dep), node_id(full), style="dashed", color="#8a8a8a",
                   arrowhead="open", **({"label": " nullable"} if nullable else {}))

    n_master = sum(1 for _, t, m in tables if not m)
    n_part = sum(1 for _, _, m in tables if m)
    refs = [(f, d) for f, _, _ in tables for d, _ in
            parsed[f]["pk_fk"] + parsed[f]["sec_fk"] if d in external]
    seam = "; ".join(f"{f}" for f, _ in refs)
    g.attr(label=f"<<br/><b>cell_registration</b>  &#183;  {n_master} master tables + "
           f"{n_part} parts  &#183;  strictly additive: nothing upstream is altered<br/>"
           '<font point-size="8.5" color="#555555">'
           "solid = foreign key inside the primary key  &#183;  dashed = secondary foreign "
           "key  &#183;  thick grey = master&#8211;part  &#183;  amber dashed = existing "
           f"production table<br/>touches the pipeline at "
           f"{'one table' if len(external) == 1 else str(len(external)) + ' tables'} only "
           f"&#8212; {', '.join(external)} &#8212; referenced by {seam}"
           "</font><br/> >", labelloc="b", fontsize="10")
    return g


def build_workflow():
    off = dict(fillcolor="#f3f1ec", color="#a79f8e")
    fil = dict(fillcolor="#fdf2d8", color="#c89f3c")
    tbl = dict(fillcolor="#d9ead3", color="#5e8b52")
    imp = dict(fillcolor="#d6e4f0", color="#436d96")

    w = graphviz.Digraph("workflow", format="png")
    w.attr(rankdir="TB", bgcolor="white", fontname="Helvetica", dpi="300",
           ranksep="0.34", nodesep="0.30", margin="0.12")
    w.attr("node", shape="box", style="filled,rounded", fontname="Helvetica",
           fontsize="10", penwidth="1.1", margin="0.14,0.08")
    w.attr("edge", fontname="Helvetica", fontsize="8.5", arrowsize="0.65", color="#5a5a5a")

    stages = [
        ("cluster_0", "  1 &#183; on a worker (Loki or ibehaveGPU &#8212; never tatchu3)", [
            ("s2p", "<<b>suite2p</b><br/><font point-size='8'>per-session segmentation</font>>",
             off, "box"),
            ("conv", "<converted_*.mat<br/><font point-size='8'>footprints per session</font>>",
             fil, "note"),
            ("cr", "<<b>CellReg</b> (MATLAB)<br/>"
             "<font point-size='8'>Sheintuch et al. 2017</font>>", off, "box"),
            ("res", "<<b>cellRegistered_*.mat</b><br/><font point-size='8'>cell_to_index_map, "
             "scores,<br/>p_same, centroids</font>>", fil, "note"),
            ("log", "<<b>logFile_*.txt</b><br/><font point-size='8'>the ONLY source of<br/>"
             "parameters and run accuracy</font>>", fil, "note"),
        ], [("s2p", "conv", ""), ("conv", "cr", ""), ("cr", "res", ""), ("cr", "log", "")]),
        ("cluster_1", "  2 &#183; register the run (manual insert)", [
            ("ps", "<<b>CellRegParamSet</b><br/><font point-size='8'>parsed from the log,<br/>"
             "hashed</font>>", tbl, "box"),
            ("run", "<<b>CellRegRun</b><br/><font point-size='8'>keyed by sha256 of the .mat,"
             "<br/>not by path</font>>", tbl, "box"),
            ("sessn", "<<b>CellRegRun.Session</b><br/><font point-size='8'>column &#8596; "
             "curation,<br/>resolved or refused</font>>", tbl, "box"),
        ], [("ps", "run", ""), ("run", "sessn", "")]),
        ("cluster_2", "  3 &#183; populate (reads the .mat)", [
            ("resu", "<<b>CellRegResult</b><br/><font point-size='8'>make() does all parsing;"
             "<br/>a bad file is recorded, not skipped</font>>", imp, "box"),
            ("cell", "<<b>.Cell</b><br/><font point-size='8'>one row per registered cell<br/>"
             "cell_index = roi_group</font>>", imp, "box"),
            ("memb", "<<b>.Member</b><br/><font point-size='8'>cell &#8596; (scan, mask)<br/>"
             "joins Segmentation.Mask directly</font>>", imp, "box"),
            ("pq", "<<b>.PairQuality</b><br/><font point-size='8'>p_same per session pair<br/>"
             "&#8776;10.7M rows lab-wide</font>>", imp, "box"),
        ], [("resu", "cell", ""), ("cell", "memb", ""), ("cell", "pq", "")]),
        ("cluster_3", "  4 &#183; use it", [
            ("query", "<<b>query</b><br/><font point-size='8'>Member &#8904; Segmentation.Mask"
             "<br/>&#8904; Activity.Trace</font>>", off, "box"),
            ("cur", "<<b>RegistrationCuration</b><br/><font point-size='8'>manual accept/reject,"
             "<br/>preserved not overwritten</font>>", tbl, "box"),
            ("proj", "<<b>RoiGroupProjection</b><br/><font point-size='8'>what roi_group WOULD "
             "become;<br/>dry_run by default</font>>", tbl, "box"),
            ("snap", "<<b>RoiGroupSnapshot</b><br/><font point-size='8'>what roi_group held "
             "before</font>>", tbl, "box"),
        ], []),
    ]
    for name, title, nodes, edges in stages:
        with w.subgraph(name=name) as s:
            s.attr(label=title, labelloc="t", fontsize="9.5", fontcolor="#6b6b6b",
                   style="rounded", color="#d8d8d8", bgcolor="#fcfcfc", margin="10")
            for nid_, lbl, style, shape in nodes:
                s.node(nid_, lbl, shape=shape, **style)
            for a, b, lbl in edges:
                s.edge(a, b, label=lbl)

    w.node("gate", "<<b>write authorisation</b><br/><font point-size='8'>Tobias, per session"
           "</font>>", fillcolor="#f7dcdc", color="#b05a5a", shape="box", style="filled")
    w.node("prod", "<<b>imaging.Segmentation.Mask.roi_group</b><br/><font point-size='8'>"
           "production &#183; smallint &#183; overwritten in place</font>>",
           shape="box", style="filled,dashed", fillcolor="#fdf2d8", color="#c89f3c",
           penwidth="1.4")

    w.edge("res", "run", label=" keyed by sha256 ")
    w.edge("log", "ps", label=" parameters ")
    w.edge("sessn", "resu")
    w.edge("memb", "query")
    w.edge("cell", "cur", style="dashed", color="#8a8a8a")
    w.edge("cur", "proj", style="dashed", color="#8a8a8a")
    w.edge("snap", "gate", style="dashed", color="#b05a5a")
    w.edge("proj", "gate", color="#b05a5a")
    w.edge("gate", "prod", color="#b05a5a", label=" only on an explicit go-ahead ")

    w.attr(label="<<br/><b>Cell registration: from the MATLAB run to a query</b><br/>"
           '<font point-size="8.5" color="#555555">'
           "amber = files and existing production tables  &#183;  green = tables this change "
           "adds  &#183;  blue = populated by make()  &#183;  red = the one path that writes "
           "production<br/>the schema is declared in code only; creating it and populating it "
           "are separate, explicitly authorised acts</font><br/> >",
           labelloc="b", fontsize="10")
    return w


def main(outdir: str = "notebooks/figures") -> int:
    src = open(SCHEMA).read()
    tables = read_tables(src)
    parsed = {full: parse_definition(read_definition(src, full.split(".")[-1], bool(master)))
              for full, _, master in tables}

    missing = [f for f, p in parsed.items() if not (p["pk_attr"] or p["pk_fk"])]
    if missing:
        raise SystemExit(f"could not parse a definition for: {missing}")

    os.makedirs(outdir, exist_ok=True)
    erd = build_erd(tables, parsed)
    erd.render(os.path.join(outdir, "erd_cell_registration"), cleanup=True)
    wf = build_workflow()
    wf.render(os.path.join(outdir, "workflow_cell_registration"), cleanup=True)

    n_master = sum(1 for _, _, m in tables if not m)
    print(f"parsed {n_master} masters + {len(tables) - n_master} parts "
          f"= {len(tables)} definitions from {os.path.relpath(SCHEMA)}")
    for f in ("erd_cell_registration.png", "workflow_cell_registration.png"):
        print("  wrote", os.path.join(outdir, f))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
