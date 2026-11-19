"""Regenerate the schema diagrams shown in README.md (``notebooks/schema_diagrams/*.svg``).

Read-only: connects with the server-side read-only switch and reads the schemas through
DataJoint virtual modules, so nothing is declared or inserted and ``adamacs.pipeline`` is
never imported. Run from a directory with ``dj_local_conf.json`` (e.g. ``~/adamacs``) in an
env with graphviz (``dot`` on PATH, python ``graphviz`` package):

    python ~/adamacs_ingest/scripts/draw_schema_diagrams.py [outdir]

Writes, into ``outdir`` (default: ``notebooks/schema_diagrams`` of this checkout):

* ``<schema>.svg``              one ``dj.Diagram`` per pipeline schema, from the live database
* ``overview.svg``              schema-level dependency graph from the live foreign keys
* ``imaging_dff0_cascade.svg``  the dF/F0 -> CASCADE tables and ROI QC (element-calcium-imaging >= 0.1.0b2)
* ``cell_registration_erd.svg`` / ``cell_registration_workflow.svg``  parsed from
  ``adamacs/schemas/cell_registration.py`` (scripts/draw_cell_registration_diagrams.py)
* ``responsiveness_erd.svg``    parsed from ``adamacs/schemas/responsiveness.py`` (not deployed yet)
* ``responsiveness_v8.svg``  parsed from the ```python blocks of ``docs/RESPONSIVENESS_SCHEMA_PROPOSAL.md``
"""
from __future__ import annotations

import collections
import os
import sys

import datajoint as dj
import graphviz

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import draw_cell_registration_diagrams as cr  # noqa: E402

PREFIX = "roselab_"

#: Schemas activated by adamacs/pipeline.py (+ the ingest-log schema), in pipeline order.
PIPELINE_SCHEMAS = ["subject", "surgery", "session", "equipment", "scan", "imaging", "train", "model",
                    "event", "trial", "behavior", "analysis", "pupil_tracking", "denoising", "mocap",
                    "virtual_markers_optitrack", "disk", "wfield", "cell_registration", "ingest"]

#: Tables of the dF/F0 -> CASCADE (+ ROI QC) focus diagram (imaging schema).
DFF0_TABLES = ["Fluorescence", "ActivityExtractionMethod", "ActivityCascadeModel", "ActivityCascadeTask",
               "DffParamSet", "CascadeRecipe", "Dff0Task", "Dff0", "Activity", "RoiQCCriterion", "RoiQC"]


def schema_diagram(name, outdir):
    sch = dj.Schema(PREFIX + name, create_schema=False, create_tables=False)
    if not sch.list_tables():
        return None
    vm = dj.create_virtual_module(name, PREFIX + name, create_schema=False, create_tables=False)
    path = os.path.join(outdir, f"{name}.svg")
    dj.Diagram(vm).save(path, format="svg")
    return path


def dff0_diagram(outdir):
    im = dj.create_virtual_module("imaging", PREFIX + "imaging", create_schema=False, create_tables=False)
    d = None
    for t in DFF0_TABLES:
        part = dj.Diagram(getattr(im, t))
        d = part if d is None else d + part
    for t in (im.Dff0.Trace, im.Activity.Trace, im.Activity.CascadeInput, im.Fluorescence.Trace, im.RoiQC.Rejected):
        d = d + dj.Diagram(t)
    path = os.path.join(outdir, "imaging_dff0_cascade.svg")
    d.save(path, format="svg")
    return path


def overview(outdir):
    conn = dj.conn()
    conn.dependencies.load()
    g_dep = conn.dependencies
    keep = {PREFIX + s for s in PIPELINE_SCHEMAS}
    edges, sizes = collections.Counter(), collections.Counter()
    for node in g_dep.nodes:
        if not node.startswith("`"):
            continue
        sch = node.split("`")[1]
        if sch in keep:
            sizes[sch] += 1
    for a, b in g_dep.edges:
        if not (a.startswith("`") and b.startswith("`")):
            continue
        sa, sb = a.split("`")[1], b.split("`")[1]
        if sa != sb and sa in keep and sb in keep:
            edges[(sa, sb)] += 1
    g = graphviz.Digraph("overview", format="svg")
    g.attr(rankdir="LR", fontname="Helvetica", ranksep="0.6", nodesep="0.25",
           label="ADAMACS (roselab_) pipeline schemas: arrows = foreign keys between schemas "
                 "(label = number of table-level references); node = schema (tables)",
           labelloc="b", fontsize="10")
    g.attr("node", shape="box", style="filled,rounded", fontname="Helvetica", fontsize="10",
           fillcolor="#eef3f8", color="#436d96")
    highlight = {PREFIX + "cell_registration": "#f4dada", PREFIX + "imaging": "#fdf2d8"}
    for s in sorted(sizes):
        g.node(s, f"{s.replace(PREFIX, '')}\n({sizes[s]})", fillcolor=highlight.get(s, "#eef3f8"))
    for (a, b), n in sorted(edges.items()):
        g.edge(a, b, label=str(n), fontsize="8", color="#6b6b6b")
    path = os.path.join(outdir, "overview")
    g.render(path, cleanup=True)
    return path + ".svg"


def parsed_erd(schema_file, title, outdir, stem):
    src = open(os.path.join(REPO, "adamacs", "schemas", schema_file)).read()
    tables = cr.read_tables(src)
    parsed = {full: cr.parse_definition(cr.read_definition(src, full.split(".")[-1], master is not None, master))
              for full, tier, master in tables}
    g = graphviz.Digraph(stem, format="svg")
    g.attr(rankdir="TB", fontname="Helvetica", ranksep="0.4", nodesep="0.26", label=title,
           labelloc="b", fontsize="10")
    g.attr("node", shape="box", style="filled,rounded", fontname="Helvetica", fontsize="10")
    own = {full for full, _, _ in tables}
    for p in parsed.values():
        for dep, _ in p["pk_fk"] + p["sec_fk"]:
            if dep != "master" and dep not in own:
                g.node(cr.node_id(dep), dep, fillcolor="#fdf2d8", color="#c89f3c", style="filled,rounded,dashed")
    for full, tier, master in tables:
        g.node(cr.node_id(full), cr.node_label(full, parsed[full], master),
               fillcolor=cr.FILL[tier], color=cr.LINE[tier])
        if master:
            g.edge(cr.node_id(master), cr.node_id(full), arrowhead="none", color="#b0b0b0", penwidth="2")
        for dep, _ in parsed[full]["pk_fk"]:
            if dep != "master":
                g.edge(cr.node_id(dep), cr.node_id(full))
        for dep, nullable in parsed[full]["sec_fk"]:
            g.edge(cr.node_id(dep), cr.node_id(full), style="dashed", color="#8a8a8a",
                   **({"label": " nullable"} if nullable else {}))
    path = os.path.join(outdir, stem)
    g.render(path, cleanup=True)
    return path + ".svg"


def proposal_diagram(doc_path, title, outdir, stem):
    """Diagram of the tables defined in the ```python blocks of a design document.

    Each class is read with its own definition in source order (part names repeat across
    masters, e.g. four ``Mask`` parts), and ``master.X`` references are resolved.
    """
    import re
    code = "\n".join(re.findall(r"```python\n(.*?)```", open(doc_path).read(), re.S))
    tables, parsed, master = [], {}, None
    for m in re.finditer(r'^( *)class (\w+)\((dj\.\w+)\):[^\n]*\n\s*definition = """(.*?)"""', code, re.M | re.S):
        indent, name, tier, defn = m.groups()
        if indent == "":
            master, full, mst = name, name, None
        else:
            full, mst, tier = f"{master}.{name}", master, "dj.Part"
        tables.append((full, tier, mst))
        parsed[full] = cr.parse_definition(defn)

    def resolve(dep, mst):
        if dep == "master":
            return mst
        return mst + "." + dep.split(".", 1)[1] if dep.startswith("master.") else dep

    g = graphviz.Digraph(stem, format="svg")
    g.attr(rankdir="TB", fontname="Helvetica", ranksep="0.45", nodesep="0.25", fontsize="10",
           labelloc="b", label=title)
    g.attr("node", shape="box", style="filled,rounded", fontname="Helvetica", fontsize="10")
    own = {f for f, _, _ in tables}
    ext = {resolve(d, mst) for f, _, mst in tables for d, _ in parsed[f]["pk_fk"] + parsed[f]["sec_fk"]} - own - {None}
    for d in sorted(ext):
        g.node(cr.node_id(d), d, fillcolor="#fdf2d8", color="#c89f3c", style="filled,rounded,dashed")
    for f, t, mst in tables:
        g.node(cr.node_id(f), cr.node_label(f, parsed[f], mst), fillcolor=cr.FILL[t], color=cr.LINE[t])
        if mst:
            g.edge(cr.node_id(mst), cr.node_id(f), arrowhead="none", color="#b0b0b0", penwidth="2")
        for dep, _ in parsed[f]["pk_fk"]:
            if dep != "master":
                g.edge(cr.node_id(resolve(dep, mst)), cr.node_id(f))
        for dep, nullable in parsed[f]["sec_fk"]:
            g.edge(cr.node_id(resolve(dep, mst)), cr.node_id(f), style="dashed", color="#8a8a8a",
                   **({"label": " nullable"} if nullable else {}))
    path = os.path.join(outdir, stem)
    g.render(path, cleanup=True)
    return path + ".svg"


def main(outdir=os.path.join(REPO, "notebooks", "schema_diagrams")):
    os.makedirs(outdir, exist_ok=True)
    dj.conn().query("SET SESSION transaction_read_only = 1")
    written = []
    for name in PIPELINE_SCHEMAS:
        p = schema_diagram(name, outdir)
        print(("wrote " + os.path.relpath(p, REPO)) if p else f"skipped {name} (no tables)")
        written.append(p)
    for fn in (overview, dff0_diagram):
        p = fn(outdir)
        print("wrote", os.path.relpath(p, REPO))
    # cell_registration: the schema's own ERD + workflow, rendered as SVG
    src = open(cr.SCHEMA).read()
    tables = cr.read_tables(src)
    parsed = {full: cr.parse_definition(cr.read_definition(src, full.split(".")[-1], master is not None, master))
              for full, tier, master in tables}
    erd = cr.build_erd(tables, parsed)
    erd.format = "svg"
    erd.render(os.path.join(outdir, "cell_registration_erd"), cleanup=True)
    wf = cr.build_workflow()
    wf.format = "svg"
    wf.render(os.path.join(outdir, "cell_registration_workflow"), cleanup=True)
    print("wrote cell_registration_erd.svg, cell_registration_workflow.svg")
    p = parsed_erd("responsiveness.py",
                   "responsiveness v8 (adamacs/schemas/responsiveness.py, piloted, not deployed); "
                   "amber dashed = pipeline tables it references", outdir, "responsiveness_erd")
    print("wrote", os.path.relpath(p, REPO))
    p = proposal_diagram(os.path.join(REPO, "docs", "RESPONSIVENESS_SCHEMA_PROPOSAL.md"),
                         "responsiveness v8 (piloted in tobiasr_responsiveness_v8): solid = FK in primary key, dashed = secondary FK, "
                         "thick grey = master-part, amber = existing production tables", outdir, "responsiveness_v8")
    print("wrote", os.path.relpath(p, REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
