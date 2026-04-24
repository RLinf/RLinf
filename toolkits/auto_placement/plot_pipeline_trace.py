import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Span:
    worker_type: str
    group_name: str | None
    rank: int | None
    pid: int | None
    hostname: str | None
    epoch: int
    stage_id: int
    chunk_step_idx: int
    op: str
    start_ns: int
    end_ns: int

    @property
    def y_label(self) -> str:
        r = "?" if self.rank is None else str(self.rank)
        return f"{self.worker_type}/r{r}"


def _iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _op_from_event(event: str) -> str | None:
    if event.endswith("_start"):
        return event[: -len("_start")]
    if event.endswith("_end"):
        return event[: -len("_end")]
    return None


def load_spans(trace_dir: str) -> list[Span]:
    files = sorted(glob.glob(os.path.join(trace_dir, "*.jsonl")))
    pending: dict[tuple, dict[str, Any]] = {}
    spans: list[Span] = []

    for fp in files:
        for rec in _iter_jsonl(fp):
            event = rec.get("event")
            if not isinstance(event, str):
                continue
            op = _op_from_event(event)
            if op is None:
                continue

            try:
                ts_ns = int(rec["ts_ns"])
                epoch = int(rec["epoch"])
                stage_id = int(rec["stage_id"])
                chunk_step_idx = int(rec["chunk_step_idx"])
            except Exception:
                continue

            worker_type = rec.get("worker_type", "unknown")
            group_name = rec.get("group_name")
            rank = rec.get("rank")
            pid = rec.get("pid")
            hostname = rec.get("hostname")
            try:
                rank = None if rank is None else int(rank)
            except Exception:
                rank = None
            try:
                pid = None if pid is None else int(pid)
            except Exception:
                pid = None

            key = (worker_type, group_name, rank, epoch, stage_id, chunk_step_idx, op)
            if event.endswith("_start"):
                pending[key] = {"ts_ns": ts_ns, "pid": pid, "hostname": hostname}
            else:
                start = pending.pop(key, None)
                if start is None:
                    continue
                start_ns = int(start["ts_ns"])
                end_ns = ts_ns
                if end_ns <= start_ns:
                    continue
                spans.append(
                    Span(
                        worker_type=str(worker_type),
                        group_name=group_name if group_name is None else str(group_name),
                        rank=rank,
                        pid=pid,
                        hostname=hostname if hostname is None else str(hostname),
                        epoch=epoch,
                        stage_id=stage_id,
                        chunk_step_idx=chunk_step_idx,
                        op=op,
                        start_ns=start_ns,
                        end_ns=end_ns,
                    )
                )

    return spans


def filter_spans(
    spans: list[Span],
    *,
    epoch: int | None,
    chunk_start: int | None,
    chunk_end: int | None,
    ranks: set[int] | None,
    worker_types: set[str] | None,
    stage_ids: set[int] | None,
) -> list[Span]:
    out: list[Span] = []
    for s in spans:
        if epoch is not None and s.epoch != epoch:
            continue
        if chunk_start is not None and s.chunk_step_idx < chunk_start:
            continue
        if chunk_end is not None and s.chunk_step_idx > chunk_end:
            continue
        if ranks is not None and (s.rank is None or s.rank not in ranks):
            continue
        if worker_types is not None and s.worker_type not in worker_types:
            continue
        if stage_ids is not None and s.stage_id not in stage_ids:
            continue
        out.append(s)
    return out


def _try_plot_plotly(
    spans: list[Span], out_path: str, title: str | None, *, x_scale: float
) -> bool:
    try:
        import pandas as pd  # type: ignore
        import plotly.express as px  # type: ignore
    except Exception:
        return False

    if not spans:
        return False

    t0 = min(s.start_ns for s in spans)
    rows = []
    for s in spans:
        stage_disp = s.stage_id + 1
        rows.append(
            {
                "Task": s.y_label,
                "Start_ms": (s.start_ns - t0) / 1e6,
                "Finish_ms": (s.end_ns - t0) / 1e6,
                "Op": s.op,
                "Label": f"c{s.chunk_step_idx}/s{stage_disp}",
                "epoch": s.epoch,
                "chunk_step_idx": s.chunk_step_idx,
                "rank": s.rank,
                "stage_id": s.stage_id,
                "worker_type": s.worker_type,
            }
        )
    df = pd.DataFrame(rows)
    fig = px.timeline(
        df,
        x_start="Start_ms",
        x_end="Finish_ms",
        y="Task",
        color="Op",
        hover_data=["worker_type", "rank", "stage_id", "epoch", "chunk_step_idx"],
        title=title or "Pipeline overlap (ms, relative)",
    )
    fig.update_traces(text=df["Label"], textposition="inside", insidetextanchor="middle")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="ms (relative)")
    # Widen figure to make inside labels readable, without changing time ordering.
    fig.update_layout(width=int(1400 * max(1.0, float(x_scale))))
    fig.write_html(out_path)
    return True


def _color_for_op(op: str) -> str:
    # Deterministic pleasant-ish palette via hashing (no deps).
    h = 0
    for ch in op:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    hue = h % 360
    return f"hsl({hue},65%,55%)"


def _plot_pure_html(
    spans: list[Span], out_path: str, title: str | None, *, x_scale: float
) -> None:
    if not spans:
        raise RuntimeError("No spans to plot after filtering.")

    t0 = min(s.start_ns for s in spans)
    t1 = max(s.end_ns for s in spans)
    total_ms = max(1.0, (t1 - t0) / 1e6)

    y_labels = sorted({s.y_label for s in spans})
    y_index = {lab: i for i, lab in enumerate(y_labels)}

    label_w = 220
    row_h = 22
    bar_h = 14
    chart_w = int(1400 * max(1.0, float(x_scale)))
    chart_h = max(120, row_h * len(y_labels) + 60)
    scale = (chart_w - label_w - 20) / total_ms

    def x_of(ns: int) -> float:
        return label_w + 10 + ((ns - t0) / 1e6) * scale

    def y_of(label: str) -> int:
        return 40 + y_index[label] * row_h

    # Build SVG bars.
    bar_elems: list[str] = []
    text_elems: list[str] = []
    for s in spans:
        x = x_of(s.start_ns)
        w = max(0.5, ((s.end_ns - s.start_ns) / 1e6) * scale)
        y = y_of(s.y_label) + (row_h - bar_h) // 2
        color = _color_for_op(s.op)
        stage_disp = s.stage_id + 1
        hover = (
            f"{s.worker_type} r{s.rank} stage{stage_disp} epoch={s.epoch} "
            f"chunk={s.chunk_step_idx} op={s.op} dur_ms={(s.end_ns - s.start_ns)/1e6:.3f}"
        )
        bar_elems.append(
            f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="{bar_h}" '
            f'rx="3" ry="3" fill="{color}"><title>{hover}</title></rect>'
        )

        label = f"c{s.chunk_step_idx}/s{stage_disp}"
        if w >= 70:
            text_elems.append(
                f'<text x="{x + w/2:.2f}" y="{y + bar_h/2 + 4}" text-anchor="middle" '
                f'font-family="monospace" font-size="11" fill="#111" opacity="0.9">{label}</text>'
            )

    # Y labels.
    label_elems = []
    for lab in y_labels:
        y = y_of(lab) + row_h - 6
        label_elems.append(
            f'<text x="{label_w - 6}" y="{y}" text-anchor="end" '
            f'font-family="monospace" font-size="12" fill="#444">{lab}</text>'
        )

    # Simple legend.
    ops = sorted({s.op for s in spans})
    legend = []
    lx, ly = label_w + 10, 18
    for op in ops:
        color = _color_for_op(op)
        legend.append(
            f'<rect x="{lx}" y="{ly - 10}" width="12" height="12" rx="2" ry="2" fill="{color}"></rect>'
        )
        legend.append(
            f'<text x="{lx + 16}" y="{ly}" font-family="sans-serif" font-size="12" fill="#333">{op}</text>'
        )
        lx += 16 + 8 * max(6, len(op))

    svg = f"""<svg width="{chart_w}" height="{chart_h}" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="{chart_w}" height="{chart_h}" fill="white"></rect>
  <text x="10" y="18" font-family="sans-serif" font-size="14" fill="#111">{(title or "Pipeline overlap (ms, relative)")}</text>
  {''.join(legend)}
  <line x1="{label_w}" y1="30" x2="{label_w}" y2="{chart_h - 10}" stroke="#ddd" stroke-width="1"></line>
  {''.join(label_elems)}
  {''.join(bar_elems)}
  {''.join(text_elems)}
</svg>"""

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{(title or "Pipeline overlap")}</title>
</head>
<body style="margin:0;padding:0;">
  <div style="padding:10px; overflow:auto;">{svg}</div>
  <div style="padding:10px; font-family:monospace; font-size:12px; color:#444;">
    time axis: ms relative to first event (t0={t0})
  </div>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def _plot_matplotlib(spans: list[Span], out_path: str, title: str | None) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    if not spans:
        raise RuntimeError("No spans to plot after filtering.")

    t0 = min(s.start_ns for s in spans)
    y_labels = sorted({s.y_label for s in spans})
    y_index = {lab: i for i, lab in enumerate(y_labels)}

    fig_h = max(3.0, 0.35 * len(y_labels))
    fig, ax = plt.subplots(figsize=(14, fig_h))

    # Group by op for legend consistency.
    ops = sorted({s.op for s in spans})
    colors = {op: None for op in ops}

    for s in spans:
        y = y_index[s.y_label]
        start_ms = (s.start_ns - t0) / 1e6
        dur_ms = (s.end_ns - s.start_ns) / 1e6
        ax.broken_barh([(start_ms, dur_ms)], (y - 0.4, 0.8), label=s.op)

    ax.set_yticks(list(range(len(y_labels))))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("ms (relative)")
    ax.set_title(title or "Pipeline overlap (ms, relative)")

    # Deduplicate legend entries.
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_h.append(h)
        uniq_l.append(l)
    if uniq_h:
        ax.legend(uniq_h, uniq_l, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


def _parse_int_set(s: str | None) -> set[int] | None:
    if not s:
        return None
    out: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def _parse_str_set(s: str | None) -> set[str] | None:
    if not s:
        return None
    out: set[str] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(part)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot pipeline overlap (Gantt) from per-worker JSONL traces."
    )
    ap.add_argument("--trace_dir", required=True, help="Directory containing *.jsonl")
    ap.add_argument(
        "--out",
        default=None,
        help="Output file path. Defaults to gantt.html (plotly) or gantt.png (mpl).",
    )
    ap.add_argument("--title", default=None, help="Chart title")
    ap.add_argument(
        "--x_scale",
        type=float,
        default=1.0,
        help="Horizontally scale time axis (keeps order/proportions) to make labels readable.",
    )

    ap.add_argument("--epoch", type=int, default=None, help="Filter: epoch index")
    ap.add_argument(
        "--chunk_start", type=int, default=None, help="Filter: chunk_step_idx >= N"
    )
    ap.add_argument(
        "--chunk_end", type=int, default=None, help="Filter: chunk_step_idx <= N"
    )
    ap.add_argument(
        "--ranks",
        default=None,
        help="Filter: ranks CSV, e.g. '0,1,2'. If omitted, keep all ranks.",
    )
    ap.add_argument(
        "--worker_types",
        default=None,
        help="Filter: worker types CSV, e.g. 'env,rollout'.",
    )
    ap.add_argument(
        "--stage_ids",
        default=None,
        help="Filter: stage_id CSV, e.g. '0,1'. If omitted, keep all stages.",
    )

    args = ap.parse_args()
    trace_dir = args.trace_dir
    spans = load_spans(trace_dir)
    spans = filter_spans(
        spans,
        epoch=args.epoch,
        chunk_start=args.chunk_start,
        chunk_end=args.chunk_end,
        ranks=_parse_int_set(args.ranks),
        worker_types=_parse_str_set(args.worker_types),
        stage_ids=_parse_int_set(args.stage_ids),
    )

    # Prefer plotly HTML if available; fallback to matplotlib PNG.
    out_path = args.out
    if out_path is None:
        out_path = os.path.join(trace_dir, "gantt.html")

    wrote = False
    if out_path.endswith(".html"):
        wrote = _try_plot_plotly(spans, out_path, args.title, x_scale=args.x_scale)
        if not wrote:
            _plot_pure_html(spans, out_path, args.title, x_scale=args.x_scale)
            wrote = True

    if not wrote:
        if not out_path.endswith(".png"):
            out_path = os.path.splitext(out_path)[0] + ".png"
        _plot_matplotlib(spans, out_path, args.title)

    print(out_path)


if __name__ == "__main__":
    main()

