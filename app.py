"""
Big 5 Leagues — Player Performance Explorer (2023–24)
======================================================
Interactive dashboard for NYU Information Visualization — Mini Project 2.

Three coordinated views with linked brushing and cross-filtering:
  V1: Goals vs Expected Goals scatter plot (position = color, age = size)
  V2: Top players by Goals + Assists horizontal bar chart (league = color)
  V3: Goals per 90 minutes strip plot by position

Dimensions encoded: Goals, xG, Position, Age, League, Goals per 90
Color palettes: ColorBrewer Set1 (position) and Dark2 (league) — colorblind-safe.

Author: Het Somaiya
Course: Information Visualization, NYU Tandon, Fall 2026
Dataset: 2023-2024 Big 5 European Soccer Player Statistics (Kaggle)
AI Assistance: Claude (Anthropic) assisted with code structure and implementation.
"""

import random
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update

# ── 1. LOAD & CLEAN DATA ──────────────────────────────────────────────────────

df = pd.read_csv("data.csv", quotechar='"')

# Standardise position: strip stray quote chars, take primary position for
# dual-listed players (e.g. "FW,MF" → "FW") so no data is dropped.
df["Position"] = df["Position"].str.strip().str.strip('"')
df["Position"] = df["Position"].str.split(",").str[0].str.strip()
df = df[df["Position"].isin(["FW", "MF", "DF", "GK"])].copy()

# Standardise competition names
df["Competition"] = df["Competition"].str.strip().str.strip('"')

# Force numeric types on all columns we use (CSV has some stored as strings)
NUMERIC_COLS = [
    "Performance_Gls", "Expected_xG", "Age",
    "Per 90 Minutes_Gls", "Performance_Ast", "Performance_G+A",
    "Expected_npxG", "Expected_xAG", "Playing Time_Min",
]
for col in NUMERIC_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows missing key numeric columns
KEY_COLS = [
    "Performance_Gls", "Expected_xG", "Age",
    "Per 90 Minutes_Gls", "Performance_Ast", "Performance_G+A",
    "Playing Time_Min",
]
df = df.dropna(subset=KEY_COLS)

# Ordinal field: age bands with a natural order (youngest → oldest)
AGE_BAND_ORDER = ["17–21", "22–25", "26–29", "30+"]
df["Age Band"] = pd.cut(
    df["Age"],
    bins=[16, 21, 25, 29, 99],
    labels=AGE_BAND_ORDER,
)
df["Age Band"] = pd.Categorical(df["Age Band"], categories=AGE_BAND_ORDER, ordered=True)

# Reset index so iloc and loc align cleanly
df = df.reset_index(drop=True)

# Shorten league names for display
LEAGUE_LABELS = {
    "eng Premier League": "Premier League",
    "es La Liga":         "La Liga",
    "de Bundesliga":      "Bundesliga",
    "it Serie A":         "Serie A",
    "fr Ligue 1":         "Ligue 1",
}
df["League"] = df["Competition"].map(LEAGUE_LABELS).fillna(df["Competition"])

AGE_MIN = int(df["Age"].min())
AGE_MAX = int(df["Age"].max())

# ── 2. COLOUR PALETTES ────────────────────────────────────────────────────────
# Position colours (ColorBrewer Set1, 4 hues): used only in V1 scatter + V3 strip
POSITION_COLORS = {
    "FW": "#e41a1c",   # red
    "MF": "#377eb8",   # blue
    "DF": "#4daf4a",   # green
    "GK": "#ff7f00",   # orange
}

# League colours: no greens or oranges — avoids cross-view collision with position palette
LEAGUE_COLORS = {
    "Premier League": "#7570b3",   # muted purple
    "La Liga":        "#e7298a",   # magenta / pink
    "Bundesliga":     "#1f78b4",   # steel blue
    "Serie A":        "#b15928",   # brown
    "Ligue 1":        "#cab2d6",   # lavender
}

# Age-band sequential ramp (light → dark) for ordinal encoding in V3
AGE_BAND_COLORS = {
    "17–21": "#c6dbef",   # lightest blue
    "22–25": "#6baed6",
    "26–29": "#2171b5",
    "30+":   "#084594",   # darkest blue
}

POSITIONS = ["FW", "MF", "DF", "GK"]
LEAGUES   = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]

# ── 3. STYLE CONSTANTS ────────────────────────────────────────────────────────
CARD_STYLE = {
    "backgroundColor": "#1e1e2e",
    "borderRadius": "10px",
    "padding": "16px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.4)",
}
FONT_FAMILY = "'Inter', 'Segoe UI', sans-serif"
AXIS_STYLE  = dict(
    color="#aaaaaa",
    gridcolor="#2e2e42",
    zerolinecolor="#3e3e5e",
    tickfont=dict(color="#aaaaaa", size=11),
    title_font=dict(color="#cccccc", size=12),
)
PLOT_BG    = "#16162a"
PAPER_BG   = "#1e1e2e"
FONT_COLOR = "#e0e0e0"

# Pre-compute stable jitter for V3 so it doesn't shift on every redraw
random.seed(42)
_jitter_map = {idx: random.uniform(-0.3, 0.3) for idx in df.index}

# ── 4. APP ────────────────────────────────────────────────────────────────────
app = Dash(__name__, title="Big 5 Soccer Explorer")
server = app.server   # exposed for Render / Gunicorn

# Fix dropdown text visibility (white-on-white issue)
app.index_string = '''<!DOCTYPE html>
<html>
<head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  .Select-value-label, .Select-value { color: #ffffff !important; }
  .Select-placeholder { color: #888888 !important; }
  .Select-input > input { color: #ffffff !important; }
  .VirtualizedSelectOption { color: #222222 !important; }
  .VirtualizedSelectFocusedOption { background-color: #4a90d9 !important; color: #ffffff !important; }
  .Select--multi .Select-value { background-color: #3e3e5e !important; border-color: #5e5e7e !important; }
  .Select--multi .Select-value-icon { border-right-color: #5e5e7e !important; }
  .Select--multi .Select-value-icon:hover { background-color: #5e5e8e !important; color: #ffffff !important; }
  .custom-dropdown { position: relative; }
  .custom-dropdown .dropdown-panel {
    display: none; position: absolute; top: 100%; left: 0; right: 0;
    z-index: 999; background-color: #2a2a3e; border: 1px solid #3e3e5e;
    border-radius: 4px; padding: 8px 12px; margin-top: 2px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
  }
  .custom-dropdown:focus-within .dropdown-panel { display: block; }
  .custom-dropdown .dropdown-panel label {
    display: block; padding: 6px 0; cursor: pointer; color: #ffffff; font-size: 13px;
  }
  .custom-dropdown .dropdown-panel input[type="checkbox"] {
    margin-right: 8px; accent-color: #4a90d9;
  }
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>'''

# ── 5. LAYOUT ─────────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={
        "backgroundColor": "#12121f",
        "minHeight": "100vh",
        "fontFamily": FONT_FAMILY,
        "padding": "24px",
    },
    children=[

        # Header
        html.Div([
            html.H1(
                "⚽  Big 5 Leagues — Player Performance Explorer (2023–24)",
                style={"color": "#ffffff", "fontSize": "22px",
                       "fontWeight": "700", "margin": "0 0 4px 0"}
            ),
            html.P(
                "Premier League · La Liga · Bundesliga · Serie A · Ligue 1  |  "
                "Select data in any view to highlight across all views.",
                style={"color": "#888888", "fontSize": "13px", "margin": 0}
            ),
        ], style={"marginBottom": "20px"}),

        # Filter bar — Gestalt: proximity groups all controls together
        html.Div([
            html.Div([
                html.Label("League", style={"color": "#aaaaaa", "fontSize": "12px",
                                            "marginBottom": "4px", "display": "block"}),
                html.Div([
                    html.Div("All leagues", id="league-display", tabIndex=0,
                             style={"backgroundColor": "#2a2a3e", "color": "#888888",
                                    "padding": "8px 12px", "borderRadius": "4px",
                                    "cursor": "pointer", "fontSize": "13px",
                                    "minHeight": "36px", "display": "flex",
                                    "alignItems": "center", "outline": "none"}),
                    html.Div([
                        dcc.Checklist(
                            id="filter-league",
                            options=[
                                {"label": " Premier League (PL)", "value": "Premier League"},
                                {"label": " La Liga (LL)",        "value": "La Liga"},
                                {"label": " Bundesliga (BL)",     "value": "Bundesliga"},
                                {"label": " Serie A (SA)",        "value": "Serie A"},
                                {"label": " Ligue 1 (L1)",        "value": "Ligue 1"},
                            ],
                            value=[],
                        ),
                    ], className="dropdown-panel"),
                ], className="custom-dropdown"),
            ], style={"flex": "1", "marginRight": "12px"}),

            html.Div([
                html.Label("Position", style={"color": "#aaaaaa", "fontSize": "12px",
                                              "marginBottom": "4px", "display": "block"}),
                html.Div([
                    html.Div("All positions", id="position-display", tabIndex=0,
                             style={"backgroundColor": "#2a2a3e", "color": "#888888",
                                    "padding": "8px 12px", "borderRadius": "4px",
                                    "cursor": "pointer", "fontSize": "13px",
                                    "minHeight": "36px", "display": "flex",
                                    "alignItems": "center", "outline": "none"}),
                    html.Div([
                        dcc.Checklist(
                            id="filter-position",
                            options=[
                                {"label": " Forwards (FW)",    "value": "FW"},
                                {"label": " Midfielders (MF)", "value": "MF"},
                                {"label": " Defenders (DF)",   "value": "DF"},
                                {"label": " Goalkeepers (GK)", "value": "GK"},
                            ],
                            value=[],
                        ),
                    ], className="dropdown-panel"),
                ], className="custom-dropdown"),
            ], style={"flex": "1", "marginRight": "12px"}),

            html.Div([
                html.Label(id="age-label",
                           style={"color": "#aaaaaa", "fontSize": "12px",
                                  "marginBottom": "4px", "display": "block"}),
                dcc.RangeSlider(
                    id="filter-age",
                    min=AGE_MIN, max=AGE_MAX,
                    value=[AGE_MIN, AGE_MAX],
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], style={"flex": "2", "marginRight": "12px"}),

            html.Div([
                html.Label("\u00a0", style={"fontSize": "12px", "display": "block",
                                            "marginBottom": "4px"}),
                html.Button("✕  Reset All", id="btn-reset", n_clicks=0,
                            style={
                                "backgroundColor": "#3e3e5e", "color": "#ffffff",
                                "border": "none", "borderRadius": "6px",
                                "padding": "8px 16px", "cursor": "pointer",
                                "fontSize": "13px", "width": "100%",
                            }),
            ], style={"flex": "0 0 110px"}),

        ], style={**CARD_STYLE, "display": "flex",
                  "alignItems": "flex-end", "marginBottom": "16px"}),

        # Top row: V1 (scatter, larger) + V2 (bar)
        html.Div([
            html.Div([
                dcc.Graph(id="v1-scatter",
                          config={"displayModeBar": False},
                          style={"height": "420px"})
            ], style={**CARD_STYLE, "flex": "3", "marginRight": "16px"}),

            html.Div([
                dcc.Graph(id="v2-bar",
                          config={"displayModeBar": False},
                          style={"height": "420px"})
            ], style={**CARD_STYLE, "flex": "2"}),
        ], style={"display": "flex", "marginBottom": "16px"}),

        # Bottom row: V3 (strip, full width)
        html.Div([
            dcc.Graph(id="v3-strip",
                      config={"displayModeBar": False},
                      style={"height": "340px"})
        ], style={**CARD_STYLE, "marginBottom": "16px"}),

        # Footer
        html.P(
            "Dataset: 2023–24 Big 5 European Soccer Player Statistics · Kaggle  |  "
            "NYU Information Visualization, Fall 2026",
            style={"color": "#555555", "fontSize": "11px", "textAlign": "center"}
        ),

        # Single shared store for selected row indices
        dcc.Store(id="store-selected", data=[]),
    ]
)

# ── 6. CALLBACKS ──────────────────────────────────────────────────────────────

# 6a. Age slider label
@app.callback(Output("age-label", "children"), Input("filter-age", "value"))
def update_age_label(val):
    return f"Age  {val[0]}–{val[1]}"


@app.callback(
    Output("league-display", "children"),
    Output("league-display", "style"),
    Input("filter-league", "value"),
)
def update_league_display(vals):
    base = {"backgroundColor": "#2a2a3e", "padding": "8px 12px",
            "borderRadius": "4px", "cursor": "pointer", "fontSize": "13px",
            "minHeight": "36px", "display": "flex", "alignItems": "center",
            "outline": "none"}
    if not vals:
        return "All leagues", {**base, "color": "#888888"}
    short = {"Premier League": "PL", "La Liga": "LL", "Bundesliga": "BL",
             "Serie A": "SA", "Ligue 1": "L1"}
    order = {"Premier League": 0, "La Liga": 1, "Bundesliga": 2, "Serie A": 3, "Ligue 1": 4}
    sorted_v = sorted(vals, key=lambda v: order.get(v, 99))
    return ", ".join(short.get(v, v) for v in sorted_v), {**base, "color": "#ffffff"}


@app.callback(
    Output("position-display", "children"),
    Output("position-display", "style"),
    Input("filter-position", "value"),
)
def update_position_display(vals):
    base = {"backgroundColor": "#2a2a3e", "padding": "8px 12px",
            "borderRadius": "4px", "cursor": "pointer", "fontSize": "13px",
            "minHeight": "36px", "display": "flex", "alignItems": "center",
            "outline": "none"}
    if not vals:
        return "All positions", {**base, "color": "#888888"}
    order = {"FW": 0, "MF": 1, "DF": 2, "GK": 3}
    sorted_v = sorted(vals, key=lambda v: order.get(v, 99))
    return ", ".join(sorted_v), {**base, "color": "#ffffff"}

# 6b. Reset button → clear filters
@app.callback(
    Output("filter-league",   "value"),
    Output("filter-position", "value"),
    Output("filter-age",      "value"),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(_):
    return [], [], [AGE_MIN, AGE_MAX]


# 6c. SINGLE callback that owns store-selected.
#     Sources: selections from V1/V3 (lasso/box), click on V2,
#              or filter changes (which clear the selection).
@app.callback(
    Output("store-selected", "data"),
    Input("v1-scatter", "selectedData"),
    Input("v2-bar",     "clickData"),
    Input("v3-strip",   "selectedData"),
    Input("filter-league",   "value"),
    Input("filter-position", "value"),
    Input("filter-age",      "value"),
    State("store-selected",  "data"),
    prevent_initial_call=True,
)
def update_selection(v1_sel, v2_click, v3_sel, leagues, positions, age_range, selected):
    ctx     = callback_context
    trigger = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    # Any filter change → clear selection
    if any(t in trigger for t in ("filter-league", "filter-position", "filter-age")):
        return []

    # V1 lasso / box select
    if "v1-scatter" in trigger:
        if v1_sel and v1_sel.get("points"):
            return [int(p["customdata"][0]) for p in v1_sel["points"]]
        return no_update


    # V2 bar click → toggle that player in/out of selection
    if "v2-bar" in trigger and v2_click:
        pts = v2_click["points"]
        clicked_player = pts[0].get("label") or pts[0].get("y")
        clicked_indices = df[df["Player"] == clicked_player].index.tolist()
        current = set(selected or [])
        if current.issuperset(clicked_indices):
            current -= set(clicked_indices)
        else:
            current |= set(clicked_indices)
        return list(current)

    # V3 lasso / box select
    if "v3-strip" in trigger:
        if v3_sel and v3_sel.get("points"):
            return [int(p["customdata"][0]) for p in v3_sel["points"]]
        return no_update

    return []


# ── helpers ───────────────────────────────────────────────────────────────────

def apply_filters(leagues, positions, age_range):
    mask = pd.Series(True, index=df.index)
    if leagues:
        mask &= df["League"].isin(leagues)
    if positions:
        mask &= df["Position"].isin(positions)
    if age_range:
        mask &= (df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])
    return mask


def opacity_list(idx_series, selected):
    """1.0 for selected (or if nothing selected); 0.12 for deselected."""
    if not selected:
        return [1.0] * len(idx_series)
    sel = set(selected)
    return [1.0 if i in sel else 0.12 for i in idx_series]


# 6d. V1 — Scatter: Goals vs xG
@app.callback(
    Output("v1-scatter", "figure"),
    Input("filter-league",   "value"),
    Input("filter-position", "value"),
    Input("filter-age",      "value"),
    Input("store-selected",  "data"),
)
def render_v1(leagues, positions, age_range, selected):
    fdf = df[apply_filters(leagues, positions, age_range)]
    fig = go.Figure()

    for pos in POSITIONS:
        sub = fdf[fdf["Position"] == pos]
        if sub.empty:
            continue
        opac = opacity_list(sub.index, selected)
        # Size encodes Minutes Played: scale 1–3420 min → 5–18 px
        sizes = sub["Playing Time_Min"].apply(
            lambda m: max(5, min(18, 5 + (m / 3420) * 13))
        ).tolist()
        fig.add_trace(go.Scatter(
            x=sub["Expected_xG"],
            y=sub["Performance_Gls"],
            mode="markers",
            name=pos,
            customdata=list(zip(
                sub.index, sub["Player"], sub["Squad"], sub["League"],
                sub["Age"], sub["Playing Time_Min"]
            )),
            marker=dict(
                color=POSITION_COLORS[pos],
                size=sizes,
                opacity=opac,
                line=dict(width=0.4, color="#ffffff"),
            ),
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "Club: %{customdata[2]}<br>"
                "League: %{customdata[3]}<br>"
                "Age: %{customdata[4]}<br>"
                "Minutes: %{customdata[5]:.0f}<br>"
                "Goals: %{y}<br>"
                "xG: %{x:.2f}<extra></extra>"
            ),
        ))

    if not fdf.empty:
        max_val = max(fdf["Expected_xG"].max(), fdf["Performance_Gls"].max(), 1)
        fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                      line=dict(color="#555577", width=1, dash="dot"))
        fig.add_annotation(x=max_val * 0.85, y=max_val * 0.78,
                      text="Goals = xG", showarrow=False,
                      font=dict(color="#666688", size=10))
        fig.add_annotation(x=max_val * 0.72, y=max_val * 0.88,
                      text="↑ Outperformed xG", showarrow=False,
                      font=dict(color="#aaaaaa", size=10))
        fig.add_annotation(x=max_val * 0.88, y=max_val * 0.68,
                      text="↓ Underperformed xG", showarrow=False,
                      font=dict(color="#aaaaaa", size=10))

    # Size legend — invisible markers shown only in legend
    for mins, sz, label in [(500, 7, "500 min"), (1500, 11, "1,500 min"), (3000, 16, "3,000 min")]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=label,
            marker=dict(size=sz, color="#aaaaaa", line=dict(width=0.4, color="#ffffff")),
            legendgroup="size", legendgrouptitle_text="Minutes Played",
            showlegend=True,
        ))

    fig.update_layout(
        title=dict(text="Goals vs Expected Goals  (dot size = minutes played)",
                   font=dict(color=FONT_COLOR, size=14), x=0, xanchor="left"),
        xaxis=dict(title="Expected Goals (xG)", **AXIS_STYLE),
        yaxis=dict(title="Goals Scored", **AXIS_STYLE),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR, family=FONT_FAMILY),
        legend=dict(font=dict(color=FONT_COLOR, size=10),
                    bgcolor="rgba(0,0,0,0)", orientation="v",
                    yanchor="top", y=0.99, xanchor="left", x=1.02),
        margin=dict(l=50, r=120, t=60, b=50),
        dragmode="select", uirevision="v1",
    )
    return fig


# 6e. V2 — Horizontal Bar: Top 15 by G+A
@app.callback(
    Output("v2-bar", "figure"),
    Input("filter-league",   "value"),
    Input("filter-position", "value"),
    Input("filter-age",      "value"),
    Input("store-selected",  "data"),
)
def render_v2(leagues, positions, age_range, selected):
    fdf = df[apply_filters(leagues, positions, age_range)]
    if fdf.empty:
        return go.Figure().update_layout(
            paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
            font=dict(color=FONT_COLOR))

    top = fdf.nlargest(15, "Performance_G+A").iloc[::-1]
    sel_set = set(selected)
    opac = [1.0 if (not selected or i in sel_set) else 0.2 for i in top.index]
    colors = [LEAGUE_COLORS.get(r["League"], "#888") for _, r in top.iterrows()]

    fig = go.Figure(go.Bar(
        x=top["Performance_G+A"],
        y=top["Player"],
        orientation="h",
        marker=dict(color=colors, opacity=opac, line=dict(width=0)),
        customdata=list(zip(
            top.index, top["Player"], top["League"],
            top["Performance_Gls"], top["Performance_Ast"]
        )),
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "League: %{customdata[2]}<br>"
            "Goals: %{customdata[3]}<br>"
            "Assists: %{customdata[4]}<br>"
            "G+A: %{x}<extra></extra>"
        ),
        showlegend=False,
    ))

    # Invisible bar traces to drive the legend (no "trace 0" ghost)
    for league, color in LEAGUE_COLORS.items():
        fig.add_trace(go.Bar(
            x=[None], y=[None],
            name=league,
            marker=dict(color=color),
            showlegend=True,
        ))

    fig.update_layout(
        title=dict(text="Top 15 Players by Goals + Assists",
                   font=dict(color=FONT_COLOR, size=14), x=0, xanchor="left"),
        xaxis=dict(title="Goals + Assists", **AXIS_STYLE),
        yaxis=dict(title="", tickfont=dict(color="#cccccc", size=10),
                   gridcolor="#2e2e42"),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR, family=FONT_FAMILY),
        legend=dict(font=dict(color=FONT_COLOR, size=10), bgcolor="rgba(0,0,0,0)",
                    orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
        margin=dict(l=120, r=20, t=60, b=50),
        bargap=0.25, uirevision="v2",
        barmode="overlay",
    )
    return fig


# 6f. V3 — Strip plot: Goals per 90 by Position
@app.callback(
    Output("v3-strip", "figure"),
    Input("filter-league",   "value"),
    Input("filter-position", "value"),
    Input("filter-age",      "value"),
    Input("store-selected",  "data"),
)
def render_v3(leagues, positions, age_range, selected):
    fdf = df[apply_filters(leagues, positions, age_range)]
    # Filter to players with ≥450 minutes to avoid inflated per-90 rates
    fdf = fdf[fdf["Playing Time_Min"] >= 450]
    fig = go.Figure()

    for pos in POSITIONS:
        sub = fdf[fdf["Position"] == pos]
        if sub.empty:
            continue

        # Dots coloured by Age Band (ordinal: sequential blue ramp light→dark)
        for band in AGE_BAND_ORDER:
            band_sub = sub[sub["Age Band"] == band]
            if band_sub.empty:
                continue
            opac  = opacity_list(band_sub.index, selected)
            pos_x = [POSITIONS.index(pos) + _jitter_map[i] for i in band_sub.index]
            fig.add_trace(go.Scatter(
                x=pos_x,
                y=band_sub["Per 90 Minutes_Gls"],
                mode="markers",
                name=band,
                legendgroup=band,
                showlegend=(pos == POSITIONS[0]),
                customdata=list(zip(
                    band_sub.index, band_sub["Player"], band_sub["Squad"],
                    band_sub["League"], band_sub["Performance_Gls"],
                    band_sub["Age"], band_sub["Playing Time_Min"]
                )),
                marker=dict(
                    color=AGE_BAND_COLORS[band],
                    size=5,
                    opacity=opac,
                    line=dict(width=0),
                ),
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "Club: %{customdata[2]}<br>"
                    "League: %{customdata[3]}<br>"
                    "Age: %{customdata[5]} (" + band + ")<br>"
                    "Minutes: %{customdata[6]:.0f}<br>"
                    "Season goals: %{customdata[4]}<br>"
                    "Goals/90: %{y:.2f}<extra></extra>"
                ),
            ))

        # Median line per position group
        med     = sub["Per 90 Minutes_Gls"].median()
        pos_idx = POSITIONS.index(pos)
        fig.add_shape(type="line",
                      x0=pos_idx - 0.35, x1=pos_idx + 0.35,
                      y0=med, y1=med,
                      line=dict(color="#ffffff", width=2.5))

    fig.update_layout(
        title=dict(
            text="Goal-Scoring Efficiency by Position  "
                 "(≥450 min · dot colour = age band · line = median)",
            font=dict(color=FONT_COLOR, size=14), x=0, xanchor="left"),
        xaxis=dict(tickvals=list(range(len(POSITIONS))), ticktext=POSITIONS,
            title="", **{**AXIS_STYLE, "gridcolor": "rgba(0,0,0,0)",
            "zerolinecolor": "rgba(0,0,0,0)"}),
        yaxis=dict(title="Goals per 90 Minutes", **AXIS_STYLE),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR, family=FONT_FAMILY),
        legend=dict(title="Age Band", font=dict(color=FONT_COLOR, size=10),
                    bgcolor="rgba(0,0,0,0)", orientation="h",
                    yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=50),
        dragmode="select", uirevision="v3",
    )
    return fig


# ── 7. RUN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
