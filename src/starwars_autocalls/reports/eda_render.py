"""Shared HTML rendering primitives for the curated EDA report and the temporal audit report.

Both reports use the same visual design system (metric grid, data tables, callouts, charts).
This module is the single place that owns that markup so a style change only needs to happen once.
"""

from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd
import plotly.io as pio

REPORT_CSS = """
:root {
  color-scheme: light;
  --bg: #f7f8fa;
  --panel: #ffffff;
  --text: #17202a;
  --muted: #5d6978;
  --line: #d9dee7;
  --accent: #1f6feb;
  --header-bg: #111827;
}
body {
  margin: 0;
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.45;
}
header { padding: 28px 36px 18px; background: var(--header-bg); color: #ffffff; }
header h1 { margin: 0 0 8px; font-size: 28px; }
header p { margin: 0; max-width: 1080px; color: #d1d5db; }
main { max-width: 1280px; margin: 0 auto; padding: 24px 28px 48px; }
section {
  margin: 0 0 28px;
  padding: 24px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}
h2 { margin: 0 0 12px; font-size: 21px; }
h3 { margin: 22px 0 10px; font-size: 16px; }
p { color: var(--muted); margin: 0 0 14px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  margin: 16px 0;
}
.metric { border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fbfcfe; }
.metric-label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.metric-value { margin-top: 4px; font-size: 22px; font-weight: 650; }
.metric-note { margin-top: 3px; color: var(--muted); font-size: 12px; }
.table-wrap { overflow-x: auto; margin: 12px 0 18px; }
table.data-table { border-collapse: collapse; width: 100%; font-size: 13px; }
table.data-table th, table.data-table td {
  border-bottom: 1px solid var(--line);
  padding: 7px 8px;
  text-align: left;
  vertical-align: top;
  white-space: nowrap;
}
table.data-table th { background: #eef2f7; font-weight: 650; }
.chart { margin: 14px 0 20px; }
.callout {
  border-left: 4px solid var(--accent);
  padding: 10px 12px;
  background: #eef6ff;
  color: #1f2937;
  margin: 14px 0;
}
.conclusions li { margin-bottom: 8px; }
"""


def fig_html(fig: Any, include_plotlyjs: bool = False) -> str:
    """Handle fig html."""
    return (
        "<div class='chart'>"
        + pio.to_html(fig, full_html=False, include_plotlyjs="cdn" if include_plotlyjs else False)
        + "</div>"
    )


def table_html(frame: pd.DataFrame, max_rows: int = 20) -> str:
    """Handle table html."""
    if frame.empty:
        return "<div class='table-wrap'><p>No data available.</p></div>"
    return (
        "<div class='table-wrap'>"
        f"{frame.head(max_rows).to_html(index=False, classes='data-table', border=0)}"
        "</div>"
    )


def metric(label: str, value: object, note: str = "") -> str:
    """Handle metric."""
    return (
        "<div class='metric'>"
        f"<div class='metric-label'>{escape(str(label))}</div>"
        f"<div class='metric-value'>{escape(str(value))}</div>"
        f"<div class='metric-note'>{escape(str(note))}</div>"
        "</div>"
    )


def html_document(title: str, header_title: str, header_subtitle: str, body: str) -> str:
    """Wrap pre-rendered section markup (`body`) in the shared page skeleton."""
    subtitle_html = f"  <p>{escape(header_subtitle)}</p>\n" if header_subtitle else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>{REPORT_CSS}</style>
</head>
<body>
<header>
  <h1>{escape(header_title)}</h1>
{subtitle_html}</header>
<main>
{body}
</main>
</body>
</html>
"""
