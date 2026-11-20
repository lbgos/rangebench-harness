"""Static results page (HTML + inline SVG, no JS) from `compare --json` output.

Usage: `python3 -m rangebench_homelab chart rows.json out.html [--title T] [--note N]...`.
No scripts so the same page fits a site with `script-src 'none'`; hover text comes from SVG <title>.
Rows: pass rate with 95% interval (and the count of error steps left out of it), then false-done, violations and trap ask rate as small
multiples in the same model order, then pass rate per session.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from .compare import Row


class ChartRow(Row):
    label: object  # model name plus any --label suffix


BAR = "#3987e5"  # reference palette series-1, dark step
W, LABEL_W, ROW_H, BAR_H = 640, 190, 30, 14


def _pct(v: object) -> str:
    return f"{v * 100:.0f}%" if isinstance(v, (int, float)) else "-"


def _hbars(
    rows: list[ChartRow], key: str, title: str, hint: str, ci: bool = False, width: int = W
) -> str:
    plot_w = width - LABEL_W - 56
    h = 34 + ROW_H * len(rows) + 22
    out = [
        f'<svg viewBox="0 0 {width} {h}" width="100%" role="img" aria-label="{html.escape(title)}">',
        f'<text x="0" y="16" class="t">{html.escape(title)}</text><text x="{width}" y="16" class="m" text-anchor="end">{html.escape(hint)}</text>',
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        x = LABEL_W + plot_w * tick
        out.append(
            f'<line x1="{x:.1f}" y1="30" x2="{x:.1f}" y2="{h - 20}" class="g"/>'
            f'<text x="{x:.1f}" y="{h - 6}" class="m" text-anchor="middle">{tick * 100:.0f}</text>'
        )
    for i, r in enumerate(rows):
        y = 34 + i * ROW_H
        v = r.get(key)
        name = html.escape(str(r["label"]))
        out.append(
            f'<text x="{LABEL_W - 10}" y="{y + BAR_H - 2}" class="l" text-anchor="end">{name}</text>'
        )
        if not isinstance(v, (int, float)):
            out.append(f'<text x="{LABEL_W + 4}" y="{y + BAR_H - 2}" class="m">no data</text>')
            continue
        bw = max(plot_w * v, 2)
        tip = f"{r['label']}: {_pct(v)}"
        errors = r.get("errors") or 0
        if ci:
            lo, hi = r["pass_ci"]
            tip += f" (95% CI {_pct(lo)}–{_pct(hi)}, {r['passed']}/{r['steps']} steps)"
            if errors:
                tip += f"; {errors} error steps left out"
        out.append(
            f"<g><title>{html.escape(tip)}</title>"
            f'<rect x="{LABEL_W}" y="{y}" width="{bw:.1f}" height="{BAR_H}" rx="2" fill="var(--bar)"/>'
        )
        end = LABEL_W + bw
        if ci:
            x1, x2 = LABEL_W + plot_w * lo, LABEL_W + plot_w * hi
            cy = y + BAR_H / 2
            out.append(
                f'<line x1="{x1:.1f}" y1="{cy}" x2="{x2:.1f}" y2="{cy}" class="w"/>'
                f'<line x1="{x1:.1f}" y1="{cy - 4}" x2="{x1:.1f}" y2="{cy + 4}" class="w"/>'
                f'<line x1="{x2:.1f}" y1="{cy - 4}" x2="{x2:.1f}" y2="{cy + 4}" class="w"/>'
            )
            end = max(end, x2)
        err = f' <tspan class="m">· {errors} err</tspan>' if ci and errors else ""
        out.append(
            f'<text x="{end + 6:.1f}" y="{y + BAR_H - 2}" class="v">{_pct(v)}{err}</text></g>'
        )
    out.append("</svg>")
    return "".join(out)


def _panels(rows: list[ChartRow], panels: tuple[tuple[str, str, str], ...]) -> str:
    """Several 0-100% bar panels side by side sharing one model-label column."""
    label_w, gap = LABEL_W, 18
    pw = (W - label_w - gap * (len(panels) - 1)) / len(panels)
    plot_w = pw - 40
    h = 50 + ROW_H * len(rows) + 22
    out = [f'<svg viewBox="0 0 {W} {h}" width="100%" role="img" aria-label="rule metrics">']
    for i, r in enumerate(rows):
        y = 50 + i * ROW_H
        out.append(
            f'<text x="{label_w - 10}" y="{y + BAR_H - 2}" class="l" text-anchor="end">{html.escape(str(r["label"]))}</text>'
        )
    for p, (key, title, hint) in enumerate(panels):
        x0 = label_w + p * (pw + gap)
        out.append(
            f'<text x="{x0}" y="16" class="t">{html.escape(title)}</text><text x="{x0}" y="32" class="m">{html.escape(hint)}</text>'
        )
        for tick in (0, 0.5, 1.0):
            x = x0 + plot_w * tick
            out.append(
                f'<line x1="{x:.1f}" y1="44" x2="{x:.1f}" y2="{h - 20}" class="g"/>'
                f'<text x="{x:.1f}" y="{h - 6}" class="m" text-anchor="middle">{tick * 100:.0f}</text>'
            )
        for i, r in enumerate(rows):
            y = 50 + i * ROW_H
            v = r.get(key)
            if not isinstance(v, (int, float)):
                out.append(f'<text x="{x0 + 4}" y="{y + BAR_H - 2}" class="m">-</text>')
                continue
            bw = max(plot_w * v, 2)
            out.append(
                f"<g><title>{html.escape(f'{r["label"]}: {title} {_pct(v)}')}</title>"
                f'<rect x="{x0}" y="{y}" width="{bw:.1f}" height="{BAR_H}" rx="2" fill="var(--bar)"/>'
                f'<text x="{x0 + bw + 5:.1f}" y="{y + BAR_H - 2}" class="v">{_pct(v)}</text></g>'
            )
    out.append("</svg>")
    return "".join(out)


def _session_table(rows: list[ChartRow]) -> str:
    sessions = sorted({s for r in rows for s in (r["sessions"] or {})})
    head = "".join(f"<th>{html.escape(str(r['label']))}</th>" for r in rows)
    body = []
    for s in sessions:
        cells = []
        for r in rows:
            v = (r["sessions"] or {}).get(s)
            shade = (
                f"background:rgba(57,135,229,{0.15 + 0.6 * v:.2f})"
                if isinstance(v, (int, float))
                else ""
            )
            cells.append(f'<td style="{shade}">{_pct(v)}</td>')
        body.append(f"<tr><th scope=row>{html.escape(s)}</th>{''.join(cells)}</tr>")
    return f"<table><thead><tr><th>session (steps 1–5)</th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render(rows: list[ChartRow], title: str, notes: list[str], embed: bool = False) -> str:
    """embed=True: no page background or padding, colors from the host theme (for an embedding frame)."""
    rows = sorted(rows, key=lambda r: -float(r.get("pass_rate") or 0))
    for r in rows:
        r.setdefault("label", r.get("model"))
    smalls = _panels(
        rows,
        (
            ("false_done", "False done", "lower is better"),
            ("violation_rate", "Rule violations", "lower is better"),
            ("trap_ask_rate", "Asked on trap steps", "higher is better"),
        ),
    )
    note_html = "".join(f"<p class=n>{html.escape(n)}</p>" for n in notes)
    theme = (
        ":root{--fg:var(--foreground);--dim:var(--muted-foreground);--line:var(--border);--bar:var(--chart-1)}body{margin:0}"
        if embed
        else ":root{color-scheme:dark;--fg:#fff;--dim:#8f8e88;--line:#262626;--bar:#3987e5}"
        "body{margin:0;padding:16px;background:#000;color:#fff;font:14px/1.4 system-ui,sans-serif}"
    )
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>
{theme}
h1{{font-size:18px;margin:0 0 4px}}.n{{color:var(--dim);margin:2px 0;font-size:13px}}
svg text{{font:12px system-ui,sans-serif}}.t{{fill:var(--fg);font-weight:600}}.l{{fill:var(--fg)}}.v{{fill:var(--fg);font-variant-numeric:tabular-nums}}
.m{{fill:var(--dim)}}.g{{stroke:var(--line);stroke-width:1}}.w{{stroke:var(--fg);stroke-width:1.5}}
.main{{max-width:{W}px;margin:16px 0}}
table{{border-collapse:collapse;margin-top:16px;font-variant-numeric:tabular-nums}}th,td{{padding:4px 10px;text-align:right;border-bottom:1px solid var(--line)}}
th[scope=row],thead th:first-child{{text-align:left;font-weight:400;color:var(--dim)}}thead th{{font-weight:600}}
.wrap{{overflow-x:auto}}
</style></head><body>
<h1>{html.escape(title)}</h1>{note_html}
<div class=main>{_hbars(rows, "pass_rate", "Steps passed", "bars: pass rate · whiskers: 95% CI", ci=True)}</div>
<div class=main>{smalls}</div>
<div class=wrap>{_session_table(rows)}</div>
</body></html>"""


def main(
    src: str, dst: str, title: str, notes: list[str], labels: list[str], embed: bool = False
) -> None:
    rows: list[ChartRow] = json.loads(Path(src).read_text(encoding="utf-8"))
    extra = dict(item.split("=", 1) for item in labels)
    for r in rows:
        if str(r.get("profile")) in extra:
            r["label"] = f"{r.get('model')} {extra[str(r.get('profile'))]}"
    Path(dst).write_text(render(rows, title, notes, embed), encoding="utf-8")
