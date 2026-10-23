"""Plan a CellReg task for one site: pick the columns, and say why each was kept.

The folder-based export made two mistakes silently, and both are why this module exists.

1. **It mixed fields of view.** CellReg takes a single ``microns_per_pixel`` for a whole run
   and only translates and rotates footprints -- it never rescales. A site can hold scans at
   different zooms (ROS-2101/sess9FUCURTR has four at 466.13 um and two at 932.26 um over the
   same 512 px), and registering those together is meaningless. ``plan_site_task`` groups the
   site's scans by field of view, keeps the largest consistent group, and reports the rest as
   excluded rather than quietly including them.
2. **It never recorded which curation a column came from.** Here the curation is chosen
   explicitly -- by default the highest ``curation_id`` per scan -- and returned so the
   caller can declare it in ``CellRegTask.Session``.

Stimulus is not a column in the database. ``scan.Scan`` carries only free-text
``scan_notes`` (often literally ``'none'``), so a stimulus is identified by its EVENTS:
gratings scans have ``event.Event.event_type`` values prefixed ``Moving_Gratings:``, one per
direction. That is the only reliable discriminator in the schema.

Nothing here writes. It returns a plan; declaring it is the caller's decision -- and it
deliberately takes a bare DataJoint ``connection`` rather than the pipeline module, because
**importing ``adamacs.pipeline`` performs an INSERT**: ``adamacs/schemas/subject.py`` inserts
a Lab row at import time. On a connection set to ``transaction_read_only`` that import fails
PART WAY, leaving ``adamacs.schemas`` without its submodule attributes, which surfaces much
later as a baffling ``ImportError: cannot import name 'cell_registration'``. Plan with SQL,
import the pipeline only when you intend to write.
"""

import collections

__all__ = ["plan_site_task", "print_plan", "stimulus_prefixes"]

DEFAULT_STIMULUS = "Moving_Gratings"


_EVENT_TABLE = {}


def _resolve(conn, schema, stem):
    """The real SQL name of a DataJoint table, which the class name does not give you.

    DataJoint prefixes by tier -- ``_`` imported/computed, ``__`` part -- so
    ``event.Event`` is ``roselab_event._event``, not ``roselab_event.event``. Guessing
    produces MissingTableError at best and an empty result at worst; ask the server.
    """
    key = (schema, stem)
    if key not in _EVENT_TABLE:
        rows = conn.query(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema=%s AND table_name REGEXP %s", (schema, f"^_*{stem}$")
        ).fetchall()
        if not rows:
            raise RuntimeError(f"no table matching {stem!r} in {schema}")
        _EVENT_TABLE[key] = rows[0][0]
    return _EVENT_TABLE[key]


def stimulus_prefixes(conn, session_id, scan_id, event_schema="roselab_event"):
    """The distinct event-type prefixes recorded for one scan, e.g. {'Moving_Gratings'}.

    This is the only reliable way to tell what a scan showed: ``scan.Scan.scan_notes`` is
    free text and is frequently the literal string 'none'.
    """
    tbl = _resolve(conn, event_schema, "event")
    rows = conn.query(
        f"SELECT DISTINCT event_type FROM `{event_schema}`.`{tbl}` "
        f"WHERE session_id=%s AND scan_id=%s", (session_id, scan_id)).fetchall()
    # Event types are not a tidy vocabulary. Bonsai VR experiments write one event per
    # frame with the texture path and timestamps inside the string
    # ("Text1On; Oddball_VRCorridor_textures_1\\grating_45_angle_31.67.png; 10; ..."),
    # so a scan can carry thousands of distinct values. Cut at the first separator and
    # clip, or the "prefixes" are unique per trial and useless for grouping -- and a
    # caller printing them floods the screen.
    out = set()
    for (e,) in rows:
        if not e:
            continue
        head = str(e).split(":")[0].split(";")[0].strip()
        out.add(head[:40])
    return out


def _fov(conn, session_id, scan_id, scan_schema="roselab_scan"):
    """(um_width, px_width) for a scan, or (None, None) when ScanInfo is absent."""
    rows = conn.query(
        f"SELECT um_width, px_width FROM `{scan_schema}`.`_scan_info__field` "
        f"WHERE session_id=%s AND scan_id=%s", (session_id, scan_id)).fetchall()
    if not rows or rows[0][0] is None:
        return (None, None)
    return (round(float(rows[0][0]), 2), int(rows[0][1]))


def plan_site_task(conn, *, subject, site, paramset_idx=10, curation="max",
                   stimulus=DEFAULT_STIMULUS, scans=None,
                   imaging_schema="roselab_imaging", session_schema="roselab_session",
                   scan_schema="roselab_scan", event_schema="roselab_event"):
    """Decide which (scan, curation) pairs should be the columns of one CellReg task.

    Parameters
    ----------
    curation : "max" | "min" | int
        Which curation of each scan to declare. "max" takes the highest ``curation_id``,
        which is the most recently curated. Whatever is chosen is RECORDED, never inferred
        later -- that is the whole point of the task table.
    stimulus : str or None
        Event-type prefix a scan must carry, e.g. "Moving_Gratings". None accepts any scan.
    scans : list[str] or None
        Explicit scan ids, in which case the stimulus filter is not applied -- an explicit
        list IS the operator's declaration. None means "every scan at the site".

    Returns
    -------
    dict with ``columns`` (ordered, each carrying the curation key and its mask count),
    ``excluded`` (each with a reason), ``microns_per_pixel`` and ``fov_groups``.
    """
    rows = conn.query(
        f"""SELECT DISTINCT c.session_id, c.scan_id, s.session_datetime
            FROM `{session_schema}`.`session_same_site` ss
            JOIN `{session_schema}`.`session` s ON s.session_id = ss.session_id
            JOIN `{imaging_schema}`.`curation` c ON c.session_id = ss.session_id
            WHERE ss.same_site_id = %s AND c.paramset_idx = %s
            ORDER BY s.session_datetime, c.scan_id""", (site, paramset_idx)).fetchall()

    cand, excluded = [], []
    for session_id, scan_id, when in rows:
        if scans is not None and scan_id not in scans:
            continue
        if scans is None and stimulus:
            prefixes = stimulus_prefixes(conn, session_id, scan_id, event_schema)
            if stimulus not in prefixes:
                excluded.append({"session_id": session_id, "scan_id": scan_id,
                                 "when": str(when)[:10],
                                 "reason": f"no {stimulus} events"
                                           + (f" (has {sorted(prefixes)[:6]}"
                                              + (f" +{len(prefixes) - 6} more" if len(prefixes) > 6
                                                 else "") + ")" if prefixes else "")})
                continue
        cids = [int(x[0]) for x in conn.query(
            f"SELECT curation_id FROM `{imaging_schema}`.`curation` "
            f"WHERE session_id=%s AND scan_id=%s AND paramset_idx=%s",
            (session_id, scan_id, paramset_idx)).fetchall()]
        if not cids:
            excluded.append({"session_id": session_id, "scan_id": scan_id,
                             "when": str(when)[:10],
                             "reason": f"no curation at paramset {paramset_idx}"})
            continue
        if curation == "max":
            cid = max(cids)
        elif curation == "min":
            cid = min(cids)
        else:
            cid = int(curation)
            if cid not in cids:
                excluded.append({"session_id": session_id, "scan_id": scan_id,
                                 "when": str(when)[:10],
                                 "reason": f"curation {cid} absent (has {sorted(cids)})"})
                continue
        n_masks = int(conn.query(
            f"SELECT COUNT(*) FROM `{imaging_schema}`.`__segmentation__mask` "
            f"WHERE session_id=%s AND scan_id=%s AND paramset_idx=%s AND curation_id=%s",
            (session_id, scan_id, paramset_idx, cid)).fetchone()[0])
        um, px = _fov(conn, session_id, scan_id, scan_schema)
        cand.append({"session_id": session_id, "scan_id": scan_id, "when": str(when)[:10],
                     "paramset_idx": paramset_idx, "curation_id": cid,
                     "n_curations": len(cids), "n_masks": n_masks,
                     "um_width": um, "px_width": px})

    # Keep the largest field-of-view group; a mixed-zoom run is not a registration.
    groups = collections.Counter((c["um_width"], c["px_width"]) for c in cand)
    keep = groups.most_common(1)[0][0] if groups else (None, None)
    columns = []
    for c in cand:
        if (c["um_width"], c["px_width"]) == keep:
            columns.append(c)
        else:
            excluded.append({**c, "reason":
                             f"field of view {c['um_width']}um/{c['px_width']}px differs from "
                             f"the site's {keep[0]}um/{keep[1]}px -- CellReg cannot rescale"})
    for i, c in enumerate(columns):
        c["column_index"] = i
    empty = [c for c in columns if c["n_masks"] == 0]
    for c in empty:
        columns.remove(c)
        excluded.append({**c, "reason": "no masks in the chosen curation"})
    for i, c in enumerate(columns):
        c["column_index"] = i

    mpp = (keep[0] / keep[1]) if keep[0] and keep[1] else None
    return {"subject": subject, "same_site_id": site, "paramset_idx": paramset_idx,
            "curation_rule": curation, "stimulus": None if scans is not None else stimulus,
            "columns": columns, "excluded": excluded,
            "microns_per_pixel": round(mpp, 4) if mpp else None,
            "fov_groups": {f"{k[0]}um/{k[1]}px": v for k, v in groups.items()},
            "n_masks_total": sum(c["n_masks"] for c in columns)}


def print_plan(plan):
    """One screen that says exactly what would be registered, and what was left out."""
    print(f"site {plan['same_site_id']} ({plan['subject']}) · paramset "
          f"{plan['paramset_idx']} · curation rule {plan['curation_rule']!r}"
          + (f" · stimulus {plan['stimulus']}" if plan["stimulus"] else " · explicit scans"))
    print(f"field of view groups: {plan['fov_groups']} -> microns_per_pixel "
          f"{plan['microns_per_pixel']}")
    print(f"\n{len(plan['columns'])} COLUMNS to register, {plan['n_masks_total']:,} masks:")
    for c in plan["columns"]:
        print(f"  {c['column_index']:>2}  {c['when']}  {c['scan_id']}  ps{c['paramset_idx']} "
              f"cur{c['curation_id']} (of {c['n_curations']})  {c['n_masks']:>6,} masks")
    if plan["excluded"]:
        print(f"\n{len(plan['excluded'])} EXCLUDED:")
        for c in plan["excluded"]:
            reason = c["reason"]
            if len(reason) > 150:                      # the event list can be thousands long
                reason = reason[:150] + " …"
            print(f"      {c.get('when','')}  {c['scan_id']}: {reason}")
    if len(plan["columns"]) < 2:
        print("\nWARNING: fewer than 2 columns -- there is nothing to register across.")
    print(f"\nmicrons_per_pixel {plan['microns_per_pixel']} is derived from "
          f"scan.ScanInfo.Field (um_width / px_width). CHECK IT against the log of any "
          f"earlier run on this site before comparing results across vintages: on "
          f"ROS-1881 ScanInfo gives 621.51/512 = 1.2139 while logFile_20260129_075612.txt "
          f"records a 615.6 um FOV, i.e. 1.2 -- about 1% apart, and it scales every "
          f"centroid distance.")


def curation_keys(plan):
    """The ordered ``imaging.Curation`` keys for CellRegTask.auto_generate_entries."""
    return [{"session_id": c["session_id"], "scan_id": c["scan_id"],
             "paramset_idx": c["paramset_idx"], "curation_id": c["curation_id"]}
            for c in plan["columns"]]
