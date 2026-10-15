import argparse
import json
import os
import sys

import pandas as pd

try:
    from plotly.offline import get_plotlyjs
except ImportError:
    sys.exit(
        "plotly is required for this script. Install it with:\n"
        "    pip install -r requirements.txt"
    )

DEMAND_COLOR = "#d62728"

DATE_CANDIDATES = ["DATE", "SHIFT_DATE"]
DEMAND_CANDIDATES = ["DEMAND", "DEMAND_TOTAL", "TOTAL_DEMAND"]


def _pick(columns, candidates, what):
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(
        f"Could not find a {what} column in {list(columns)}. "
        f"Expected one of: {', '.join(candidates)}"
    )


def load_daily(demand_path: str) -> pd.DataFrame:
    if not os.path.exists(demand_path):
        raise FileNotFoundError(f"Demand file not found: {demand_path}")

    df = pd.read_csv(demand_path)
    date_col = _pick(df.columns, DATE_CANDIDATES, "date")
    demand_col = _pick(df.columns, DEMAND_CANDIDATES, "demand")

    df = df.rename(columns={date_col: "DATE", demand_col: "DEMAND"})
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.strftime("%Y-%m-%d")
    df["DEMAND"] = pd.to_numeric(df["DEMAND"], errors="coerce").fillna(0)

    daily = df.groupby("DATE", as_index=False)["DEMAND"].sum().sort_values("DATE")
    daily["DEMAND"] = daily["DEMAND"].astype(int)
    return daily


def build_html(daily: pd.DataFrame, title: str) -> str:
    records = [
        {"date": r.DATE, "demand": int(r.DEMAND)} for r in daily.itertuples(index=False)
    ]
    plotlyjs = get_plotlyjs()
    data_json = json.dumps(records, separators=(",", ":"))

    total = int(daily["DEMAND"].sum())
    days = int(daily["DEMAND"].count())
    avg = int(round(total / days)) if days else 0
    peak = int(daily["DEMAND"].max()) if days else 0
    peak_day = daily.loc[daily["DEMAND"].idxmax(), "DATE"] if days else "-"

    html = TEMPLATE
    html = html.replace("__TITLE__", title)
    html = html.replace("__PLOTLYJS__", plotlyjs)
    html = html.replace("__DATA_JSON__", data_json)
    html = html.replace("__DEMAND_COLOR__", DEMAND_COLOR)
    html = html.replace("__TOTAL__", f"{total:,}")
    html = html.replace("__DAYS__", f"{days:,}")
    html = html.replace("__AVG__", f"{avg:,}")
    html = html.replace("__PEAK__", f"{peak:,}")
    html = html.replace("__PEAK_DAY__", peak_day)
    return html


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         color: #1c1c1e; background: #f5f5f7; }
  header { padding: 16px 24px; background: #fff; border-bottom: 1px solid #e2e2e6; }
  header h1 { margin: 0 0 4px; font-size: 18px; }
  header p { margin: 0; color: #6b6b70; font-size: 13px; }
  .layout { display: flex; gap: 16px; padding: 16px 24px; align-items: stretch; }
  .controls { width: 240px; flex: 0 0 240px; display: flex; flex-direction: column; gap: 16px; }
  .panel { background: #fff; border: 1px solid #e2e2e6; border-radius: 10px; padding: 12px; }
  .panel h2 { margin: 0 0 8px; font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: #6b6b70; }
  label.radio { display: flex; align-items: center; gap: 6px; font-size: 13px; margin: 4px 0; cursor: pointer; }
  .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .stat b { display: block; font-size: 16px; }
  .stat span { font-size: 11px; color: #6b6b70; }
  .legend { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; }
  .swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
  #chart { flex: 1; min-width: 0; height: calc(100vh - 160px); min-height: 520px;
           background: #fff; border: 1px solid #e2e2e6; border-radius: 10px; padding: 8px; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <p>Demand per day across __DAYS__ days &mdash; drag the slider under the chart to zoom into a date range.</p>
</header>
<div class="layout">
  <aside class="controls">
    <div class="panel">
      <h2>View</h2>
      <label class="radio"><input type="radio" name="view" value="cumulative" checked/> Accumulated (running total)</label>
      <label class="radio"><input type="radio" name="view" value="daily"/> Daily (per day)</label>
    </div>
    <div class="panel">
      <h2>Summary</h2>
      <div class="stats">
        <div class="stat"><b>__TOTAL__</b><span>Total demand</span></div>
        <div class="stat"><b>__AVG__</b><span>Avg / day</span></div>
        <div class="stat"><b>__PEAK__</b><span>Peak day</span></div>
        <div class="stat"><b>__DAYS__</b><span>Days</span></div>
      </div>
      <p style="margin:10px 0 0; font-size:11px; color:#6b6b70;">Peak on __PEAK_DAY__</p>
    </div>
    <div class="panel">
      <span class="legend"><i class="swatch" style="background:__DEMAND_COLOR__"></i>Demand</span>
    </div>
  </aside>
  <main id="chart"></main>
</div>

<script>__PLOTLYJS__</script>
<script>
const DATA = __DATA_JSON__;
const DEMAND_COLOR = "__DEMAND_COLOR__";
const state = { cumulative: true };

function compute() {
  const dates = DATA.map(d => d.date);
  const demand = [];
  let cum = 0;
  for (const d of DATA) {
    if (state.cumulative) { cum += d.demand; demand.push(cum); }
    else { demand.push(d.demand); }
  }
  return { dates, demand };
}

function render() {
  const s = compute();
  const trace = {
    x: s.dates, y: s.demand, name: "Demand", mode: "lines",
    line: { color: DEMAND_COLOR, width: 2 },
    fill: "tozeroy", fillcolor: "rgba(214,39,40,0.08)",
    hovertemplate: "%{x}<br>Demand: %{y:,}<extra></extra>"
  };
  const layout = {
    margin: { l: 60, r: 20, t: 20, b: 40 },
    hovermode: "x unified",
    xaxis: {
      type: "date", title: "Date", gridcolor: "#ececf0",
      rangeslider: { visible: true, thickness: 0.08 }
    },
    yaxis: { title: state.cumulative ? "Accumulated demand" : "Demand per day", gridcolor: "#ececf0", rangemode: "tozero" },
    showlegend: false,
    paper_bgcolor: "#fff", plot_bgcolor: "#fff"
  };
  Plotly.react("chart", [trace], layout, { responsive: true, displaylogo: false });
}

document.addEventListener("DOMContentLoaded", () => {
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
        description="Build an interactive HTML plot of (accumulated) demand per day from a demand CSV."
    )
    parser.add_argument(
        "--demand", type=str, default="data/Demand.csv",
        help="Path to a demand CSV (date + demand columns; Demand.csv is summed per day)",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/demand_plot.html",
        help="Path to write the interactive HTML file",
    )
    parser.add_argument("--title", type=str, default="Demand per Day", help="Chart title")
    parser.add_argument("--open", action="store_true", help="Open the HTML file in the default browser")
    args = parser.parse_args()

    daily = load_daily(args.demand)
    html = build_html(daily, args.title)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Wrote interactive plot to {args.output}")
    print(f"  days        : {daily['DEMAND'].count()}")
    print(f"  total demand: {int(daily['DEMAND'].sum()):,}")

    if args.open:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
