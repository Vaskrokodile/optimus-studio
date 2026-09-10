"""Build a localhost timeline viewer for all NYC three.js attempts.

Scans E:/iloptimus-home/learning/*, filters to the NYC query, copies every
artifact (baseline / adapted / framework) into a served directory, and emits
an index.html that lays them out as a chronological timeline with metadata,
scores, events, and live iframe previews.
"""
from __future__ import annotations

import json
import os
import shutil
import html
from datetime import datetime, timezone
from pathlib import Path

LEARNING = Path(r"E:/iloptimus-home/learning")
OUT = Path(r"E:/optimusstudio/iloptimus/nyc_timeline")
ARTIFACTS = OUT / "artifacts"
NYC_QUERY_TOKEN = "New York City"

VARIANT_LABELS = {
    "baseline": "Baseline",
    "adapted": "Adapted",
    "framework": "Framework",
}


def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return None


def collect_attempts():
    attempts = []
    for d in sorted(LEARNING.iterdir()):
        if not d.is_dir():
            continue
        sess = load_json(d / "session.json")
        if not sess:
            continue
        q = sess.get("query", "") or ""
        if NYC_QUERY_TOKEN.lower() not in q.lower():
            continue
        artifacts = {}
        for variant in ("baseline", "adapted", "framework"):
            idx = d / variant / "index.html"
            if idx.exists():
                artifacts[variant] = idx
        attempts.append({"id": d.name, "session": sess, "artifacts": artifacts, "dir": d})
    attempts.sort(key=lambda a: a["session"].get("created_at", 0))
    return attempts


def copy_artifacts(attempts):
    if ARTIFACTS.exists():
        shutil.rmtree(ARTIFACTS)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for a in attempts:
        for variant, src in a["artifacts"].items():
            dst_dir = ARTIFACTS / a["id"] / variant
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / "index.html")
            # copy screenshot if present
            shot = src.parent / "runtime.png"
            if shot.exists():
                shutil.copy2(shot, dst_dir / "runtime.png")


def fmt_ts(ts):
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def fmt_dur(a, b):
    if not a or not b:
        return ""
    s = int(b - a)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s//60}m {s%60}s"
    return f"{s//3600}h {(s%3600)//60}m"


def eval_summary(ev):
    if not ev:
        return None
    fs = ev.get("feature_scores") or {}
    gates = ev.get("hard_gates") or {}
    passed_gates = sum(1 for v in gates.values() if v)
    return {
        "score": ev.get("score"),
        "bytes": ev.get("bytes"),
        "lines": ev.get("lines"),
        "passed": ev.get("passed"),
        "features": fs,
        "gates_passed": passed_gates,
        "gates_total": len(gates),
        "gates": gates,
    }


def render_events(events):
    if not events:
        return "<p class='muted'>no events</p>"
    rows = []
    for e in events:
        seq = e.get("sequence", "")
        stage = html.escape(str(e.get("stage", "")))
        msg = html.escape(str(e.get("message", "")))
        rows.append(f"<li><span class='seq'>#{seq}</span> <span class='stage'>{stage}</span> {msg}</li>")
    return "<ol class='events'>" + "".join(rows) + "</ol>"


def render_attempt(idx, a):
    s = a["session"]
    created = s.get("created_at")
    updated = s.get("updated_at")
    be = eval_summary(s.get("baseline_evaluation"))
    ae = eval_summary(s.get("adapted_evaluation"))
    fe = eval_summary(s.get("framework_evaluation"))
    acc = s.get("acceptance") or {}

    artifacts = a["artifacts"]
    variant_blocks = []
    for variant in ("baseline", "adapted", "framework"):
        if variant not in artifacts:
            continue
        ev = {"baseline": be, "adapted": ae, "framework": fe}[variant]
        score = f"{ev['score']:.3f}" if ev and ev["score"] is not None else "—"
        gates = f"{ev['gates_passed']}/{ev['gates_total']}" if ev else "—"
        passed = ev["passed"] if ev else None
        badge = ""
        if passed is True:
            badge = "<span class='badge pass'>PASS</span>"
        elif passed is False:
            badge = "<span class='badge fail'>FAIL</span>"
        feat = ""
        if ev and ev["features"]:
            feat = " ".join(
                f"<span class='feat'>{k}:{v:.1f}</span>" for k, v in ev["features"].items()
            )
        src = f"artifacts/{a['id']}/{variant}/index.html"
        shot = f"artifacts/{a['id']}/{variant}/runtime.png"
        variant_blocks.append(f"""
        <div class="variant">
          <div class="variant-head">
            <h4>{VARIANT_LABELS[variant]}</h4>
            <span class="score">score {score}</span>
            <span class="gates">gates {gates}</span>
            {badge}
          </div>
          <div class="feat-row">{feat}</div>
          <div class="preview">
            <iframe loading="lazy" src="{src}" title="{a['id']} {variant}"></iframe>
          </div>
          <div class="shot-link"><a href="{shot}" target="_blank">runtime screenshot</a> · <a href="{src}" target="_blank">open artifact</a></div>
        </div>""")

    events_html = render_events(s.get("events") or [])
    acc_html = ""
    if acc:
        acc_html = (
            f"<div class='acc'>acceptance: accepted={acc.get('accepted')} "
            f"baseline={acc.get('baseline_score')} adapted={acc.get('adapted_score')} "
            f"improvement={acc.get('improvement')} "
            f"<span class='muted'>{html.escape(str(acc.get('reason','')))}</span></div>"
        )

    return f"""
    <section class="attempt" id="a{idx}">
      <div class="attempt-meta">
        <div class="num">#{idx}</div>
        <div class="meta-main">
          <h3>{a['id']}</h3>
          <div class="kv">
            <span><b>model</b> {html.escape(str(s.get('model_id','')))}</span>
            <span><b>status</b> {html.escape(str(s.get('status','')))}</span>
            <span><b>stage</b> {html.escape(str(s.get('stage','')))}</span>
            <span><b>progress</b> {s.get('progress')}</span>
            <span><b>created</b> {fmt_ts(created)}</span>
            <span><b>updated</b> {fmt_ts(updated)}</span>
            <span><b>duration</b> {fmt_dur(created, updated)}</span>
            <span><b>events</b> {len(s.get('events') or [])}</span>
          </div>
          {acc_html}
        </div>
      </div>
      <div class="variants">{''.join(variant_blocks)}</div>
      <details class="events-wrap"><summary>events ({len(s.get('events') or [])})</summary>{events_html}</details>
    </section>"""


def render_index(attempts):
    n = len(attempts)
    completed = sum(1 for a in attempts if a["session"].get("status") == "completed")
    best = None
    best_score = -1
    for a in attempts:
        for ev in (a["session"].get("baseline_evaluation"), a["session"].get("adapted_evaluation"), a["session"].get("framework_evaluation")):
            if ev and ev.get("score") is not None and ev["score"] > best_score:
                best_score = ev["score"]
                best = (a["id"], ev.get("score"))
    body = "".join(render_attempt(i + 1, a) for i, a in enumerate(attempts))
    nav = "".join(
        f"<a href='#a{i+1}'>#{i+1}</a>" for i in range(n)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NYC three.js attempts — timeline</title>
<style>
  :root {{
    --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3;
    --muted:#8b949e; --accent:#58a6ff; --pass:#3fb950; --fail:#f85149;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
         background:var(--bg); color:var(--text); }}
  header {{ position:sticky; top:0; z-index:10; background:#0d1117ee;
            backdrop-filter:blur(8px); border-bottom:1px solid var(--border);
            padding:12px 20px; }}
  header h1 {{ margin:0 0 4px; font-size:18px; }}
  header .stats {{ color:var(--muted); font-size:12px; }}
  nav {{ margin-top:8px; display:flex; flex-wrap:wrap; gap:6px; }}
  nav a {{ color:var(--accent); text-decoration:none; padding:2px 8px;
           border:1px solid var(--border); border-radius:6px; font-size:11px; }}
  nav a:hover {{ background:var(--panel); }}
  main {{ max-width:1600px; margin:0 auto; padding:20px; }}
  .attempt {{ background:var(--panel); border:1px solid var(--border);
              border-radius:10px; padding:18px; margin-bottom:22px; }}
  .attempt-meta {{ display:flex; gap:14px; margin-bottom:14px; }}
  .num {{ font-size:22px; font-weight:700; color:var(--accent); min-width:42px; }}
  .meta-main h3 {{ margin:0 0 6px; font-family:ui-monospace,Consolas,monospace;
                   font-size:13px; color:var(--muted); }}
  .kv {{ display:flex; flex-wrap:wrap; gap:6px 14px; font-size:12px; }}
  .kv b {{ color:var(--muted); font-weight:500; margin-right:3px; }}
  .acc {{ margin-top:6px; font-size:12px; }}
  .muted {{ color:var(--muted); }}
  .variants {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr));
               gap:14px; }}
  .variant {{ background:#0d1117; border:1px solid var(--border);
              border-radius:8px; padding:10px; }}
  .variant-head {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap;
                   margin-bottom:6px; }}
  .variant-head h4 {{ margin:0; font-size:13px; }}
  .score,.gates {{ font-family:ui-monospace,Consolas,monospace; font-size:12px;
                   color:var(--text); }}
  .badge {{ font-size:10px; padding:1px 6px; border-radius:4px; font-weight:700; }}
  .badge.pass {{ background:#1b3a23; color:var(--pass); }}
  .badge.fail {{ background:#3a1b1b; color:var(--fail); }}
  .feat-row {{ margin-bottom:6px; }}
  .feat {{ font-family:ui-monospace,Consolas,monospace; font-size:10px;
           background:#21262d; padding:1px 5px; border-radius:4px; margin-right:4px; }}
  .preview {{ position:relative; width:100%; padding-top:62%; border-radius:6px;
              overflow:hidden; border:1px solid var(--border); background:#000; }}
  .preview iframe {{ position:absolute; inset:0; width:100%; height:100%;
                     border:0; }}
  .shot-link {{ margin-top:6px; font-size:11px; }}
  .shot-link a {{ color:var(--accent); text-decoration:none; margin-right:8px; }}
  .events-wrap {{ margin-top:14px; }}
  .events-wrap summary {{ cursor:pointer; color:var(--muted); font-size:12px; }}
  .events {{ margin:8px 0 0; padding-left:18px; font-size:11px; }}
  .events li {{ margin-bottom:2px; }}
  .seq {{ color:var(--muted); font-family:ui-monospace,Consolas,monospace; }}
  .stage {{ color:var(--accent); font-family:ui-monospace,Consolas,monospace; }}
</style>
</head>
<body>
<header>
  <h1>NYC three.js attempts — timeline</h1>
  <div class="stats">{n} attempts · {completed} completed ·
    {f"best score {best[1]:.3f} ({best[0]})" if best else "no scores"}</div>
  <nav>{nav}</nav>
</header>
<main>
{body}
</main>
</body>
</html>"""


def main():
    attempts = collect_attempts()
    print(f"found {len(attempts)} NYC attempts")
    copy_artifacts(attempts)
    html_out = render_index(attempts)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(html_out, encoding="utf-8")
    print(f"wrote {OUT / 'index.html'}")
    print(f"artifacts in {ARTIFACTS}")
    for a in attempts:
        print(f"  {a['id']}  created={fmt_ts(a['session'].get('created_at'))}  variants={list(a['artifacts'])}")


if __name__ == "__main__":
    main()
