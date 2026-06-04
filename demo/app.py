"""
Analyzing Demographic Biases — interactive poster demo.

Finding: across three transformer AES models (XLNet, RoBERTa, Longformer),
adversarial debiasing (Gradient Reversal Layer, "GRL") fails to reduce
demographic bias without losing scoring quality.

Everything shown is precomputed in demo_predictions.csv — no model inference.
Single page, mobile-friendly, free-tier compatible.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

# --------------------------------------------------------------------------- #
# Paths / load
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "demo_predictions.csv")
# Precomputed aggregate SMD table (5 attrs x human/baseline/GRL), computed once from
# the full test set and shipped alongside the app so the deployed CSV stays small
# (HF Spaces enforces a 10 MiB/file limit). If absent (e.g. local dev with the full
# CSV present), the SMD table is computed from the CSV instead — identical values.
SMD_AGG_PATH = os.path.join(HERE, "smd_aggregate.csv")
CHART_PATH = os.path.join("/tmp", "smd_chart.png")

MODELS = ["XLNet", "RoBERTa", "Longformer"]
GITHUB_URL = "https://github.com/SCU-CSEN346/Analyzing-Demographic-Biases"

df_raw = pd.read_csv(CSV_PATH, dtype={"essay_id": str})
# The demo is about GRL only — projection rows never appear in the UI or chart.
df = df_raw[df_raw["method"] != "projection"].copy()
df["full_text"] = df["full_text"].fillna("")

# --------------------------------------------------------------------------- #
# Full-text safety net (local dev). On HF Spaces full_text is already baked into
# the CSV, so this block does nothing. Only triggers if raw corpora are present.
# --------------------------------------------------------------------------- #
def _backfill_full_text(frame: pd.DataFrame) -> pd.DataFrame:
    cur = frame[frame["in_curated_set"] == True]  # noqa: E712
    need = cur.groupby("essay_id")["full_text"].first()
    missing = [e for e, t in need.items() if not str(t).strip()]
    if not missing:
        return frame
    candidates = {
        "PERSUADE": ([
            "projects/nlp/DATA/PERSUADE/test/persuade_corpus_2.0_test.csv",
            "projects/nlp/DATA/PERSUADE/train/persuade_corpus_2.0_train.csv",
        ], "essay_id_comp"),
        "ASAP": ([
            "projects/nlp/DATA/ASAP/test/ASAP_2_Final_github_test.csv",
            "projects/nlp/DATA/ASAP/train/ASAP_2_Final_github_train.csv",
        ], "essay_id"),
    }
    ft_map = {}
    for corpus, (paths, key) in candidates.items():
        ids = set(cur[cur["corpus"] == corpus]["essay_id"])
        for rel in paths:
            p = os.path.join(HERE, rel)
            if not os.path.exists(p) or not ids:
                continue
            try:
                d = pd.read_csv(p, usecols=[key, "full_text"], dtype=str, low_memory=False)
            except Exception:
                continue
            d = d.dropna(subset=[key]).drop_duplicates(subset=[key])
            m = dict(zip(d[key], d["full_text"]))
            for e in list(ids):
                if e in m and str(m[e]).strip():
                    ft_map[e] = m[e]
                    ids.discard(e)
    if ft_map:
        mask = (frame["in_curated_set"] == True) & (frame["essay_id"].isin(ft_map))  # noqa: E712
        frame.loc[mask, "full_text"] = frame.loc[mask, "essay_id"].map(ft_map)
    return frame

df = _backfill_full_text(df)

# --------------------------------------------------------------------------- #
# Curated essays: metadata, predictions, dropdown labels
# --------------------------------------------------------------------------- #
RACE_SHORT = {
    "black/african american": "Black",
    "white": "White",
    "hispanic/latino": "Hispanic",
    "asian/pacific islander": "Asian",
    "two or more races/other": "Two or more",
    "american indian/alaskan native": "Native",
}


def _clean_prompt(p: str) -> str:
    return str(p).strip().strip('"').strip()


def _short_race(r) -> str:
    if not isinstance(r, str) or not r.strip():
        return "—"
    return RACE_SHORT.get(r.strip().lower(), r.strip().title())


def _label(meta: dict) -> str:
    g = {"m": "M", "f": "F"}.get(str(meta["gender"]).strip().lower(), "?")
    race = _short_race(meta["race_ethnicity"])
    tags = []
    if str(meta["ell_status"]).strip().lower() == "yes":
        tags.append("ELL")
    if str(meta["student_disability_status"]).strip().lower().startswith("identified"):
        tags.append("disability")
    if str(meta["economically_disadvantaged"]).strip().lower() == "economically disadvantaged":
        tags.append("SES")
    base = f"[{meta['corpus']}] {_clean_prompt(meta['prompt_name'])} — {g}, {race}"
    if tags:
        base += ", " + ", ".join(tags)
    if len(base) > 80:  # keep labels phone-friendly
        base = base[:77].rstrip(", ") + "…"
    return base


curated = df[df["in_curated_set"] == True].copy()  # noqa: E712

META = {}      # essay_id -> metadata dict (incl. full_text, true_score)
PREDS = {}     # essay_id -> {model: {"baseline": x, "GRL": y}}
for eid, grp in curated.groupby("essay_id"):
    first = grp.iloc[0]
    META[eid] = {
        "essay_id": eid,
        "corpus": first["corpus"],
        "prompt_name": first["prompt_name"],
        "gender": first["gender"],
        "race_ethnicity": first["race_ethnicity"],
        "ell_status": first["ell_status"],
        "economically_disadvantaged": first["economically_disadvantaged"],
        "student_disability_status": first["student_disability_status"],
        "true_score": float(first["true_score"]),
        "full_text": first["full_text"] if str(first["full_text"]).strip() else "(essay text unavailable)",
    }
    pm = {}
    for m in MODELS:
        sub = grp[grp["model"] == m]
        b = sub[sub["method"] == "baseline"]["pred_score"]
        gg = sub[sub["method"] == "GRL"]["pred_score"]
        pm[m] = {
            "baseline": float(b.iloc[0]) if len(b) else float("nan"),
            "GRL": float(gg.iloc[0]) if len(gg) else float("nan"),
        }
    PREDS[eid] = pm

# Sort: PERSUADE first, then ASAP; then by essay_id. Dropdown choices = (label, value).
_corpus_rank = {"PERSUADE": 0, "ASAP": 1}
_ordered_ids = sorted(META, key=lambda e: (_corpus_rank.get(META[e]["corpus"], 9), e))
CHOICES = [(_label(META[e]), e) for e in _ordered_ids]

# --------------------------------------------------------------------------- #
# Aggregate SMD chart (computed once, saved to PNG)
# --------------------------------------------------------------------------- #
# (focal_value, reference_value) per attribute, on the lowercased column.
ATTRS = [
    ("Gender",     "gender",                     "f",                              "m"),
    ("Race (B/W)", "race_ethnicity",             "black/african american",         "white"),
    ("ELL",        "ell_status",                 "yes",                            "no"),
    ("Low SES",    "economically_disadvantaged", "economically disadvantaged",     "not economically disadvantaged"),
    ("Disability", "student_disability_status",  "identified as having disability", "not identified as having disability"),
]


def _smd(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    denom = np.sqrt((sa ** 2 + sb ** 2) / 2.0)
    if denom == 0 or np.isnan(denom):
        return np.nan
    return abs((a.mean() - b.mean()) / denom)


def _col_lower(frame, col):
    return frame[col].astype(str).str.strip().str.lower()


def compute_smd_table(frame):
    """Returns dict attr -> {'human':, 'baseline':, 'GRL':} of |SMD|, averaged over models."""
    out = {}
    essays = frame.drop_duplicates("essay_id")            # human score is per-essay
    base = frame[frame["method"] == "baseline"]
    grl = frame[frame["method"] == "GRL"]
    for name, col, foc, ref in ATTRS:
        # Human reference
        ev = _col_lower(essays, col)
        h = _smd(essays.loc[ev == foc, "true_score"], essays.loc[ev == ref, "true_score"])
        # Model baseline / GRL: average across the 3 models
        b_vals, g_vals = [], []
        for m in MODELS:
            bm = base[base["model"] == m]
            gm = grl[grl["model"] == m]
            bl = _col_lower(bm, col)
            gl = _col_lower(gm, col)
            b_vals.append(_smd(bm.loc[bl == foc, "pred_score"], bm.loc[bl == ref, "pred_score"]))
            g_vals.append(_smd(gm.loc[gl == foc, "pred_score"], gm.loc[gl == ref, "pred_score"]))
        out[name] = {
            "human": h,
            "baseline": np.nanmean(b_vals),
            "GRL": np.nanmean(g_vals),
        }
    return out


def load_smd_table():
    """Prefer the shipped precomputed aggregate; else compute from the CSV."""
    if os.path.exists(SMD_AGG_PATH):
        agg = pd.read_csv(SMD_AGG_PATH)
        return {
            row["attribute"]: {
                "human": float(row["human"]),
                "baseline": float(row["baseline"]),
                "GRL": float(row["GRL"]),
            }
            for _, row in agg.iterrows()
        }
    return compute_smd_table(df)


def build_chart(smd, path):
    names = [a[0] for a in ATTRS]
    human = [smd[n]["human"] for n in names]
    base = [smd[n]["baseline"] for n in names]
    grl = [smd[n]["GRL"] for n in names]

    C_HUMAN, C_BASE, C_GRL = "#8B0000", "#E08A3C", "#F2C94C"
    y = np.arange(len(names))
    bh = 0.26
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    ax.barh(y + bh, human, height=bh, color=C_HUMAN, label="Human scores")
    ax.barh(y, base, height=bh, color=C_BASE, label="Model — baseline")
    ax.barh(y - bh, grl, height=bh, color=C_GRL, edgecolor="#caa400", label="Model — after GRL")

    for yi, vals in zip(y, zip(human, base, grl)):
        for off, v in zip((bh, 0, -bh), vals):
            if not np.isnan(v):
                ax.text(v + 0.012, yi + off, f"{v:.2f}", va="center", fontsize=7.5, color="#333")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    allv = [v for v in (human + base + grl) if not np.isnan(v)]
    ax.set_xlim(0, max(0.7, (max(allv) if allv else 0.7) + 0.10))
    ax.set_xlabel("|SMD|  —  standardized gap between groups", fontsize=10)
    ax.set_title("Standardized score gap per demographic attribute", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return smd


SMD_TABLE = load_smd_table()
build_chart(SMD_TABLE, CHART_PATH)

# --------------------------------------------------------------------------- #
# Per-essay view + caption
# --------------------------------------------------------------------------- #
def _caption(base_avg, grl_avg, delta_avg, true_score):
    if delta_avg < 0.2:
        return "GRL changed the prediction by less than 0.2 points across all three models."
    if abs(grl_avg - true_score) > abs(base_avg - true_score):
        return "GRL moved the prediction further from the human score."
    if base_avg < true_score - 0.3:
        return "The models under-predicted this essay. GRL didn't fix it."
    if base_avg > true_score + 0.3:
        return "The models over-predicted this essay. GRL didn't fix it."
    return "Predictions shifted slightly, but the pattern repeats — see the aggregate chart below."


def essay_view(essay_id):
    meta = META[essay_id]
    pm = PREDS[essay_id]
    true_score = meta["true_score"]

    text = meta["full_text"]

    score_md = (
        f"### 🧑‍🏫 Human (teacher) score: **{true_score:.1f}**\n"
        f"<span style='color:#666'>Scores are on a 1–6 scale (higher = better)</span>"
    )

    table = pd.DataFrame(
        {
            "Model": MODELS,
            "Baseline": [round(pm[m]["baseline"], 2) for m in MODELS],
            "After GRL": [round(pm[m]["GRL"], 2) for m in MODELS],
        }
    )

    base_vals = [pm[m]["baseline"] for m in MODELS]
    grl_vals = [pm[m]["GRL"] for m in MODELS]
    base_avg = float(np.mean(base_vals))
    grl_avg = float(np.mean(grl_vals))
    delta_avg = float(np.mean([abs(g - b) for g, b in zip(grl_vals, base_vals)]))
    caption = f"**{_caption(base_avg, grl_avg, delta_avg, true_score)}**"

    return text, score_md, table, caption


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
CSS = """
.gradio-container {max-width: 820px !important; margin: 0 auto !important;}
#headline h1 {margin-bottom: 2px; line-height: 1.2;}
#headline p {color:#555; margin-top: 0;}
#smd-img img {width: 100%; height: auto;}
footer {visibility: hidden;}
"""

_default_id = CHOICES[0][1]
_d_text, _d_score, _d_table, _d_caption = essay_view(_default_id)

with gr.Blocks(css=CSS, title="Analyzing Demographic Biases — Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# When debiasing doesn't hold.\n"
        "Pick an essay below — see what happens.\n\n"
        "<span style='color:#888; font-size:0.9em'>This demo focuses on GRL debiasing; "
        "for orthogonal projection results see our paper.</span>",
        elem_id="headline",
    )

    picker = gr.Dropdown(
        choices=CHOICES,
        value=_default_id,
        label="📄 Choose an essay",
        interactive=True,
    )

    essay_box = gr.Textbox(
        value=_d_text,
        label="Essay",
        lines=8,
        max_lines=16,
        interactive=False,
        show_copy_button=False,
    )

    score_box = gr.Markdown(_d_score)

    table_box = gr.Dataframe(
        value=_d_table,
        headers=["Model", "Baseline", "After GRL"],
        datatype=["str", "number", "number"],
        interactive=False,
        row_count=(3, "fixed"),
        col_count=(3, "fixed"),
        label="Predicted scores (baseline vs. after GRL debiasing)",
        wrap=True,
    )

    caption_box = gr.Markdown(_d_caption)

    gr.Markdown("---\n## Across the full test set")
    gr.Image(
        value=CHART_PATH,
        label=None,
        show_label=False,
        show_download_button=False,
        container=False,
        elem_id="smd-img",
    )
    gr.Markdown(
        "Across every attribute, GRL doesn't close the gap between human and model scoring. "
        "The model **amplifies** the demographic gap (taller bars than human), and the "
        "after-GRL bar stays about as tall as the baseline."
    )

    gr.Markdown(
        "---\n"
        "**Authors:** Rina Li, Tom Ngo, Karthik Tamil  \n"
        "**Advisor:** Dr. Oana Ignat · Santa Clara University  \n"
        f"**Code:** [{GITHUB_URL.replace('https://', '')}]({GITHUB_URL})"
    )

    picker.change(
        fn=essay_view,
        inputs=picker,
        outputs=[essay_box, score_box, table_box, caption_box],
    )


if __name__ == "__main__":
    demo.launch()