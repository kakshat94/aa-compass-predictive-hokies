"""
Plot coverage data.
"""

import argparse
import json
import os
import sys
import logging

import pandas as pd

try:
    from plotly.offline import get_plotlyjs
except ImportError:
    sys.exit(
        "plotly is required for this script. Install it with:\n"
        "    pip install -r requirements.txt"
    )

REQUIRED_COLUMNS = ["SHIFT_DATE", "WORK_CENTER", "SHIFT_TYPE", "DEMAND", "SUPPLY"]

DEMAND_COLOR = "#d62728"
SUPPLY_COLOR = "#1f77b4"

SHIFT_ORDER = ["MORNING", "EVENING", "MIDNIGHT"]


def load_daily(report_path: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Coverage report not found: {report_path}")

    df = pd.read_csv(report_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{report_path} is missing required column(s): {', '.join(missing)}. "
            f"Found: {', '.join(df.columns)}"
        )

    df["SHIFT_DATE"] = pd.to_datetime(df["SHIFT_DATE"]).dt.strftime("%Y-%m-%d")
    for col in ("DEMAND", "SUPPLY"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    daily = df.groupby(["SHIFT_DATE", "WORK_CENTER", "SHIFT_TYPE"], as_index=False)[
        ["DEMAND", "SUPPLY"]
    ].sum()

    work_centers = sorted(daily["WORK_CENTER"].unique().tolist())
    shift_types = sorted(
        daily["SHIFT_TYPE"].unique().tolist(),
        key=lambda s: (
            SHIFT_ORDER.index(s) if s in SHIFT_ORDER else len(SHIFT_ORDER),
            s,
        ),
    )
    return daily, work_centers, shift_types


def build_html(daily: pd.DataFrame, work_centers, shift_types, title: str) -> str:
    records = [
        {
            "date": r.SHIFT_DATE,
            "wc": r.WORK_CENTER,
            "st": r.SHIFT_TYPE,
            "demand": int(r.DEMAND),
            "supply": int(r.SUPPLY),
        }
        for r in daily.itertuples(index=False)
    ]

    plotlyjs = get_plotlyjs()
    data_json = json.dumps(records, separators=(",", ":"))
    wc_json = json.dumps(work_centers)
    st_json = json.dumps(shift_types)
    totals = {
        "demand": int(daily["DEMAND"].sum()),
        "supply": int(daily["SUPPLY"].sum()),
        "days": int(daily["SHIFT_DATE"].nunique()),
    }

    html = TEMPLATE
    html = html.replace("__TITLE__", title)
    html = html.replace("__PLOTLYJS__", plotlyjs)
    html = html.replace("__DATA_JSON__", data_json)
    html = html.replace("__WC_JSON__", wc_json)
    html = html.replace("__ST_JSON__", st_json)
    html = html.replace("__DEMAND_COLOR__", DEMAND_COLOR)
    html = html.replace("__SUPPLY_COLOR__", SUPPLY_COLOR)
    html = html.replace("__TOTAL_DEMAND__", f"{totals['demand']:,}")
    html = html.replace("__TOTAL_SUPPLY__", f"{totals['supply']:,}")
    html = html.replace("__TOTAL_DAYS__", f"{totals['days']:,}")
    return html


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<style>
  :root { --demand: __DEMAND_COLOR__; --supply: __SUPPLY_COLOR__; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         color: #1c1c1e; background: #f5f5f7; }
  header { padding: 16px 24px; background: #fff; border-bottom: 1px solid #e2e2e6; }
  header h1 { margin: 0 0 4px; font-size: 18px; }
  header p { margin: 0; color: #6b6b70; font-size: 13px; }
  .layout { display: flex; gap: 16px; padding: 16px 24px; align-items: stretch; }
  .controls { width: 260px; flex: 0 0 260px; display: flex; flex-direction: column; gap: 16px; }
  .panel { background: #fff; border: 1px solid #e2e2e6; border-radius: 10px; padding: 12px; }
  .panel h2 { margin: 0 0 8px; font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: #6b6b70; }
  select[multiple] { width: 100%; border: 1px solid #d0d0d5; border-radius: 8px; padding: 4px; font-size: 13px; }
  .btnrow { display: flex; gap: 6px; margin-top: 8px; }
  .btnrow button { flex: 1; padding: 5px 8px; font-size: 12px; border: 1px solid #d0d0d5; border-radius: 6px;
                   background: #fff; cursor: pointer; }
  .btnrow button:hover { background: #f0f0f3; }
  label.radio { display: flex; align-items: center; gap: 6px; font-size: 13px; margin: 4px 0; cursor: pointer; }
  .stats { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 6px; }
  .stat b { display: block; font-size: 16px; }
  .stat span { font-size: 11px; color: #6b6b70; }
  #chart { flex: 1; min-width: 0; height: calc(100vh - 160px); min-height: 520px;
           background: #fff; border: 1px solid #e2e2e6; border-radius: 10px; padding: 8px; }
  .legend { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; margin-right: 14px; }
  .swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <p>Accumulated demand vs. supply per day &mdash; filter by work center and shift type. Dataset spans __TOTAL_DAYS__ days.</p>
</header>
<div class="layout">
  <aside class="controls">
    <div class="panel">
      <h2>Work center</h2>
      <select id="wc" multiple size="8"></select>
      <div class="btnrow">
        <button data-target="wc" data-action="all">All</button>
        <button data-target="wc" data-action="none">None</button>
      </div>
    </div>
    <div class="panel">
      <h2>Shift type</h2>
      <select id="st" multiple size="4"></select>
      <div class="btnrow">
        <button data-target="st" data-action="all">All</button>
        <button data-target="st" data-action="none">None</button>
      </div>
    </div>
    <div class="panel">
      <h2>View</h2>
      <label class="radio"><input type="radio" name="view" value="cumulative" checked/> Accumulated (running total)</label>
      <label class="radio"><input type="radio" name="view" value="daily"/> Daily (per day)</label>
    </div>
    <div class="panel">
      <h2>Selection summary</h2>
      <div class="stats">
        <div class="stat"><b id="s-demand">0</b><span>Demand</span></div>
        <div class="stat"><b id="s-supply">0</b><span>Supply</span></div>
        <div class="stat"><b id="s-under" style="color:__DEMAND_COLOR__">0</b><span>Under</span></div>
        <div class="stat"><b id="s-over" style="color:__SUPPLY_COLOR__">0</b><span>Over</span></div>
      </div>
      <div style="margin-top:10px">
        <span class="legend"><i class="swatch" style="background:__DEMAND_COLOR__"></i>Demand</span>
        <span class="legend"><i class="swatch" style="background:__SUPPLY_COLOR__"></i>Supply</span>
      </div>
    </div>
  </aside>
  <main id="chart"></main>
</div>

<script>__PLOTLYJS__</script>
<script>
const DATA = __DATA_JSON__;
const ALL_WC = __WC_JSON__;
const ALL_ST = __ST_JSON__;
const DEMAND_COLOR = "__DEMAND_COLOR__";
const SUPPLY_COLOR = "__SUPPLY_COLOR__";

const state = { wc: new Set(ALL_WC), st: new Set(ALL_ST), cumulative: true };

function fillSelect(el, values) {
  el.innerHTML = "";
  for (const v of values) {
    const opt = document.createElement("option");
    opt.value = v; opt.textContent = v; opt.selected = true;
    el.appendChild(opt);
  }
}

function readSelection(el) {
  return new Set(Array.from(el.selectedOptions).map(o => o.value));
}

function fmt(n) { return n.toLocaleString(); }

function compute() {
  const rows = DATA.filter(r => state.wc.has(r.wc) && state.st.has(r.st));
  const byDate = new Map();
  let demand = 0, supply = 0, under = 0, over = 0;
  for (const r of rows) {
    const agg = byDate.get(r.date) || { demand: 0, supply: 0 };
    agg.demand += r.demand; agg.supply += r.supply;
    byDate.set(r.date, agg);
    demand += r.demand; supply += r.supply;
    const d = Math.max(0, r.demand - r.supply);
    const o = Math.max(0, r.supply - r.demand);
    under += d; over += o;
  }
  const dates = Array.from(byDate.keys()).sort();
  const demandSeries = [], supplySeries = [];
  let cumD = 0, cumS = 0;
  for (const dt of dates) {
    const v = byDate.get(dt);
    if (state.cumulative) { cumD += v.demand; cumS += v.supply; }
    else { cumD = v.demand; cumS = v.supply; }
    demandSeries.push(cumD); supplySeries.push(cumS);
  }
  return { dates, demandSeries, supplySeries, demand, supply, under, over };
}

function render() {
  const s = compute();
  document.getElementById("s-demand").textContent = fmt(s.demand);
  document.getElementById("s-supply").textContent = fmt(s.supply);
  document.getElementById("s-under").textContent = fmt(s.under);
  document.getElementById("s-over").textContent = fmt(s.over);

  const demandTrace = {
    x: s.dates, y: s.demandSeries, name: "Demand", mode: "lines",
    line: { color: DEMAND_COLOR, width: 2 }, hovertemplate: "%{x}<br>Demand: %{y:,}<extra></extra>"
  };
  const supplyTrace = {
    x: s.dates, y: s.supplySeries, name: "Supply", mode: "lines",
    line: { color: SUPPLY_COLOR, width: 2 }, hovertemplate: "%{x}<br>Supply: %{y:,}<extra></extra>"
  };
  const layout = {
    margin: { l: 60, r: 20, t: 20, b: 50 },
    hovermode: "x unified",
    xaxis: { type: "date", title: "Shift date", gridcolor: "#ececf0" },
    yaxis: { title: state.cumulative ? "Accumulated shifts" : "Shifts per day", gridcolor: "#ececf0", rangemode: "tozero" },
    legend: { orientation: "h", y: 1.08 },
    paper_bgcolor: "#fff", plot_bgcolor: "#fff"
  };
  Plotly.react("chart", [demandTrace, supplyTrace], layout, { responsive: true, displaylogo: false });
}

function syncFromSelects() {
  state.wc = readSelection(document.getElementById("wc"));
  state.st = readSelection(document.getElementById("st"));
  render();
}

document.addEventListener("DOMContentLoaded", () => {
  fillSelect(document.getElementById("wc"), ALL_WC);
  fillSelect(document.getElementById("st"), ALL_ST);

  document.getElementById("wc").addEventListener("change", syncFromSelects);
  document.getElementById("st").addEventListener("change", syncFromSelects);

  document.querySelectorAll(".btnrow button").forEach(btn => {
    btn.addEventListener("click", () => {
      const el = document.getElementById(btn.dataset.target);
      const on = btn.dataset.action === "all";
      Array.from(el.options).forEach(o => o.selected = on);
      syncFromSelects();
    });
  });

  document.querySelectorAll('input[name="view"]').forEach(r => {
    r.addEventListener("change", e => { state.cumulative = e.target.value === "cumulative"; render(); });
  });

  render();
});
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Build an interactive HTML plot of accumulated demand vs. supply from a coverage report CSV."
    )
    parser.add_argument(
        "--report",
        type=str,
        default="outputs/coverage_report.csv",
        help="Path to the coverage report CSV produced by common/coverage_report.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/coverage_plot.html",
        help="Path to write the interactive HTML file",
    )
    parser.add_argument(
        "--title", type=str, default="Demand vs Supply Coverage", help="Chart title"
    )
    parser.add_argument(
        "--open", action="store_true", help="Open the HTML file in the default browser"
    )
    args = parser.parse_args()

    daily, work_centers, shift_types = load_daily(args.report)
    html = build_html(daily, work_centers, shift_types, args.title)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    logging.info(f"Wrote interactive plot to {args.output}")
    logging.info(f"  work centers: {', '.join(work_centers)}")
    logging.info(f"  shift types : {', '.join(shift_types)}")
    logging.info(f"  days        : {daily['SHIFT_DATE'].nunique()}")

    if args.open:
        import webbrowser

        webbrowser.open(f"file://{os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
