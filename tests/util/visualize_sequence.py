from typing import Optional, Any, Dict, Iterable, List
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from tests.util.entities import (
    Event,
    EventSpecialCode,
    get_midi_instrument_name_from_midi_instrument_code,
)


def _events_to_df(events: Iterable[Event]) -> pd.DataFrame:
    events_list = list(events)
    if len(events_list) == 0:
        raise ValueError("events is empty")

    norm: List[tuple[Any, ...]] = []
    for i, e in enumerate(events_list):
        # idx here is "sequence index order" (input order).
        norm.append(
            (
                # idx
                i,
                # start
                e.absolute_time,
                # duration
                e.midi_duration(),
                # program
                e.midi_instrument(),
                # note
                e.midi_note(),
                # note_name
                e.note().to_name(),
                # special_code
                e.special_code,
                # is_control
                e.is_control,
                # original_idx_in_token_seq
                e.original_idx_in_token_seq,
                # rel_time_start
                e.midi_time(),
            )
        )

    df = pd.DataFrame(
        norm,
        columns=[
            "idx",
            "start",
            "duration",
            "program",
            "note",
            "note_name",
            "special_code",
            "is_control",
            "original_idx_in_token_seq",
            "rel_time_start",
        ],
    )
    if (df["duration"] < 0).any():
        raise ValueError("Found negative durations.")
    df["end"] = df["start"] + df["duration"]
    return df


def _build_program_palette(programs: list[int]) -> Dict[int, str]:
    palette = (
        qualitative.Plotly
        + qualitative.D3
        + qualitative.G10
        + qualitative.T10
        + qualitative.Dark24
        + qualitative.Light24
    )
    programs_sorted = sorted(programs)
    return {p: palette[i % len(palette)] for i, p in enumerate(programs_sorted)}


def _add_boundaries_to_subplot(
    fig: go.Figure,
    df_boundaries: pd.DataFrame,
    delta: float,
    time_resolution: int,
    context_length: int,
) -> None:
    """
    Adds boundary lines + annotations to the *top* subplot only.
    Uses subplot-aware add_shape/add_annotation so it doesn't span the index subplot.
    """
    dfb = df_boundaries.sort_values(by="start", kind="mergesort")
    prev_end = 0
    for row in dfb.to_dict("records"):
        special_code = row["special_code"]
        if special_code == EventSpecialCode.SEQ_SEPARATION_TOKENS:
            fig.add_shape(
                type="line",
                x0=row["start"],
                x1=row["start"],
                y0=0,
                y1=1,
                xref="x",
                yref="y domain",
                layer="above",
                row=1,
                col=1,
                line={"width": 1.2},
            )
            # top of the graph
            sep_idx = row["original_idx_in_token_seq"]
            fig.add_annotation(
                x=row["start"],
                y=1,
                xref="x",
                yref="y domain",
                text=f"SEP (idx={sep_idx:,})",
                showarrow=False,
                yanchor="bottom",
                yshift=-14,
                row=1,
                col=1,
            )
        elif special_code == EventSpecialCode.AUTOREGRESSIVE_TOKEN:
            fig.add_shape(
                type="line",
                x0=row["start"],
                x1=row["start"],
                y0=0,
                y1=1,
                xref="x",
                yref="y domain",
                layer="above",
                row=1,
                col=1,
                line={"dash": "dash", "width": 0.7},
            )
            # top of the graph
            fig.add_annotation(
                x=row["start"],
                y=1,
                xref="x",
                yref="y domain",
                text="AR",
                showarrow=False,
                yanchor="bottom",
                yshift=0,
                row=1,
                col=1,
            )
        elif special_code == EventSpecialCode.ANTICIPATION_TOKEN:
            fig.add_shape(
                type="line",
                x0=row["start"],
                x1=row["start"],
                y0=0,
                y1=1,
                xref="x",
                yref="y domain",
                layer="above",
                row=1,
                col=1,
                line={"dash": "dash", "width": 0.7},
            )
            # top of the graph
            fig.add_annotation(
                x=row["start"],
                y=1,
                xref="x",
                yref="y domain",
                text="ANTI",
                showarrow=False,
                yanchor="bottom",
                yshift=0,
                row=1,
                col=1,
            )

        if row["original_idx_in_token_seq"] % context_length == 0:
            # context windows on bottom graph, HORIZONTAL lines
            fig.add_shape(
                type="line",
                x0=prev_end,
                x1=prev_end,
                y0=0,
                y1=1,
                xref="x",
                yref="y domain",
                layer="above",
                row=1,
                col=1,
                line={"dash": "dash", "width": 0.7},
            )
            fig.add_shape(
                type="line",
                x0=0,
                x1=1,
                y0=row["idx"],
                y1=row["idx"],
                xref="x domain",
                yref="y",
                layer="above",
                row=2,
                col=1,
                line={"dash": "dot", "width": 0.7},
            )
            fig.add_annotation(
                # the far right of the graph is 1 (?)
                x=1,
                y=row["idx"],
                xref="x domain",
                yref="y",
                text="context",
                showarrow=False,
                yanchor="bottom",
                yshift=0,
                row=2,
                col=1,
            )

        prev_end = max(prev_end, row["end"])

    first_delta_sec_in_ticks = int(delta * time_resolution)
    fig.add_shape(
        type="line",
        x0=first_delta_sec_in_ticks,
        x1=first_delta_sec_in_ticks,
        y0=0,
        y1=1,
        xref="x",
        yref="y domain",
        layer="above",
        row=2,
        col=1,
        line={"dash": "dash", "width": 0.7},
    )
    fig.add_annotation(
        x=first_delta_sec_in_ticks,
        y=1,
        xref="x",
        yref="y domain",
        text="Delta Seconds",
        showarrow=False,
        yanchor="bottom",
        yshift=0,
        row=2,
        col=1,
    )


def plot_pianoroll_with_index_timeline(
    events: list[Event],
    delta: float,
    time_resolution: int,
    *,
    title: str = "MIDI Piano Roll",
    height: int = 1000,
    y_jitter: float = 0.18,
    x_jitter: int = 2,
) -> go.Figure:
    df = _events_to_df(events)
    is_boundary = ~(
        (df["special_code"] == EventSpecialCode.TYPICAL_EVENT)
        | (df["special_code"] == EventSpecialCode.TICK)
    )
    df_boundaries = df.loc[is_boundary].copy()
    df_notes = df.loc[~is_boundary].copy()

    if df_notes.empty:
        raise ValueError(
            "No note events remain after filtering special-code boundary events."
        )

    programs_sorted = sorted(df_notes["program"].unique().tolist())
    program_to_color = _build_program_palette(programs_sorted)

    # Subplots: top = roll, bottom = index timeline
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.6, 0.4],
    )

    # --- Row 1: Piano roll (one trace per program) ---
    hover_template_roll = (
        "Program: %{meta.program} (%{meta.program_name})<br>"
        "Note: %{customdata[0]} (%{customdata[1]})<br>"
        "Start (absolute): %{customdata[2]} ticks<br>"
        "Start (relative): %{customdata[8]} ticks<br>"
        "End (absolute): %{customdata[3]} ticks<br>"
        "Dur: %{customdata[4]} ticks<br>"
        "Event Idx: %{customdata[5]}<br>"
        "Token Idx: %{customdata[7]}<br>"
        "Is Control: %{customdata[6]}"
        "<extra></extra>"
    )

    # Small vertical offsets to separate overlapping notes
    # magnitude chosen to be visually clear but musically negligible
    program_offsets = {
        p: (
            (i - (len(programs_sorted) - 1) / 2) * y_jitter,
            (i - (len(programs_sorted) - 1) / 2) * x_jitter,
        )
        for i, p in enumerate(programs_sorted)
    }

    for midi_program_code in programs_sorted:
        sub = df_notes[df_notes["program"] == midi_program_code].sort_values(
            ["start", "note"], kind="mergesort"
        )
        midi_program_name = get_midi_instrument_name_from_midi_instrument_code(
            midi_program_code
        )

        # jitter so we can distinguish between different instruments that
        # happen at the exact same time on the exact same note
        y_off, x_off = program_offsets[midi_program_code]

        in_legend = False
        for label, sub2, dash, width, opacity in [
            ("notes", sub[sub["is_control"] == False], "solid", 8, 0.5),
            ("control", sub[sub["is_control"] == True], "dot", 8, 1.0),
        ]:
            if sub2.empty:
                continue

            x: list[Optional[float]] = []
            y: list[Optional[float]] = []
            custom: list[Optional[list[Any]]] = []

            for (
                idx,
                start,
                end,
                note,
                note_name,
                dur,
                is_control,
                original_idx_in_token_seq,
                rel_time_start,
            ) in sub2[
                [
                    "idx",
                    "start",
                    "end",
                    "note",
                    "note_name",
                    "duration",
                    "is_control",
                    "original_idx_in_token_seq",
                    "rel_time_start",
                ]
            ].itertuples(index=False):
                x0 = int(start) + x_off
                x1 = int(end) + x_off
                yv = int(note) + y_off

                if is_control:
                    yv += 0.2

                x.extend([x0, x1, None])
                y.extend([yv, yv, None])

                row = [
                    note_name,
                    int(note),
                    int(start),
                    int(end),
                    int(dur),
                    int(idx),
                    bool(is_control),
                    int(original_idx_in_token_seq),
                    int(rel_time_start),
                ]
                custom.extend([row, row, None])

            marker_settings = {
                130: {
                    "symbol": "triangle-sw",
                    "size": 14,
                }
            }
            fig.add_trace(
                go.Scattergl(
                    x=x,
                    y=y,
                    customdata=custom,
                    mode="lines+markers",
                    line=dict(
                        width=6,
                        dash=dash,
                        color=program_to_color[midi_program_code],
                    ),
                    marker=(
                        marker_settings.get(midi_program_code)
                        or (
                            dict(
                                # defaults
                                symbol="line-ns",
                                size=20,
                            )
                        )
                    ),
                    opacity=opacity,
                    name=f"{midi_program_code}: {midi_program_name}",
                    meta={
                        "program": int(midi_program_code),
                        "program_name": midi_program_name,
                    },
                    showlegend=(True if not in_legend else False),
                    hovertemplate=hover_template_roll,
                    connectgaps=False,
                    legendgroup=f"prog_{midi_program_code}",
                    xaxis="x2",
                ),
                row=1,
                col=1,
            )
            in_legend = True

    df_seq = df_notes.sort_values(["idx"], kind="mergesort").copy()
    df_seq["x_mid"] = (df_seq["start"] + df_seq["end"]) / 2.0

    fig.add_trace(
        go.Scatter(
            x=df_seq["x_mid"].to_list(),
            y=df_seq["note"].to_list(),
            mode="lines+markers+text",
            name="Sequence order path",
            visible="legendonly",  # starts off; user can toggle on in legend
            line=dict(width=1),  # keep subtle; increase if desired
            marker=dict(
                symbol="arrow",
                size=11,
                angleref="previous",
                standoff=4,
            ),
            text=df_seq["idx"].astype(int).astype(str).to_list(),
            textposition="top center",
            textfont=dict(size=10),
            showlegend=True,
            xaxis="x2",
        ),
        row=1,
        col=1,
    )
    # dummy trace to keep rangeslider alive on xaxis
    t0 = float(df["start"].min())
    t1 = float(df["end"].max())

    fig.add_trace(
        go.Scatter(
            x=[t0, t1],
            y=[0, 0],
            mode="lines",
            line=dict(width=0),
            opacity=0.0,
            hoverinfo="skip",
            showlegend=False,
            xaxis="x",  # stays on x
            yaxis="y",
        ),
        row=1,
        col=1,
    )

    # adds all the context lines and delta seconds
    settings = events[0].settings
    _add_boundaries_to_subplot(
        fig,
        df_boundaries,
        delta,
        time_resolution,
        settings.context_size,
    )

    # --- Row 2: Index timeline (time vs idx) ---
    # We keep the same per-program coloring to make correlations easier.
    hover_template_idx = (
        "Start (absolute): %{x} ticks<br>"
        "Start (relative): %{customdata[7]} ticks<br>"
        "Program: %{customdata[1]} (%{customdata[2]})<br>"
        "Note: %{customdata[3]} (%{customdata[4]})<br>"
        "Is Control: %{customdata[5]}<br>"
        "Event Idx: %{customdata[0]}<br>"
        "Token Idx: %{customdata[6]}<br>"
        "<extra></extra>"
    )

    for midi_program_code in programs_sorted:
        sub = df_notes[df_notes["program"] == midi_program_code].copy()
        midi_program_name = get_midi_instrument_name_from_midi_instrument_code(
            midi_program_code
        )
        sub = sub.sort_values(["start", "idx"], kind="mergesort")

        fig.add_trace(
            go.Scatter(
                x=sub["start"].astype(int).tolist(),
                y=sub["idx"].astype(int).tolist(),
                mode="markers",
                marker=dict(
                    size=9,
                    color=program_to_color[midi_program_code],
                    symbol=[
                        "star" if is_ctrl else "square-open"
                        for is_ctrl in sub["is_control"].astype(bool).tolist()
                    ],
                    opacity=0.6,
                ),
                name=f"{midi_program_code}: {midi_program_name} (idx)",
                showlegend=False,  # keep legend focused on instruments in top plot
                legendgroup=f"prog_{midi_program_code}",
                customdata=np.stack(
                    [
                        sub["idx"].astype(int).to_numpy(),
                        sub["program"].astype(int).to_numpy(),
                        np.array([midi_program_name] * len(sub), dtype=object),
                        sub["note_name"].astype(str).to_numpy(),
                        sub["note"].astype(int).to_numpy(),
                        sub["is_control"].astype(bool).to_numpy(),
                        sub["original_idx_in_token_seq"].astype(int).to_numpy(),
                        sub["rel_time_start"].astype(int).to_numpy(),
                    ],
                    axis=1,
                ),
                hovertemplate=hover_template_idx,
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(
        title_text="sequence idx",
        row=2,
        col=1,
        constrain="range",
        minallowed=0,
        maxallowed=len(events),
    )
    fig.update_yaxes(
        title_text="MIDI note", row=1, col=1, constrain="range", minallowed=0
    )
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=70, r=30, t=60, b=60),
        hovermode="closest",
        legend_title_text="Program",
    )
    fig.update_layout(uirevision=True)
    return fig


def write_fig_html(
    fig: go.Figure, save_to_path: Path, *, auto_open: bool = True
) -> None:
    assert save_to_path.suffix == ".html"
    fig.write_html(
        str(save_to_path.absolute()),
        auto_open=auto_open,
    )


def get_figure_and_open(
    events: list[Event],
    delta: float,
    time_resolution: int,
    path: Path,
    auto_open: bool = True,
) -> None:
    fig1 = plot_pianoroll_with_index_timeline(events, delta, time_resolution)
    write_fig_html(fig1, path, auto_open=auto_open)
