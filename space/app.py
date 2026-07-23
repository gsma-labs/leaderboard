"""Open Telco Leaderboard — Gradio Space.

Replicates the website's dark-themed leaderboard using server-rendered HTML
inside gr.HTML components. Loads live data from the GSMA/leaderboard HF dataset.
"""

from __future__ import annotations

import ast
import base64
import html as html_mod
import time
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import pandas as pd
from datasets import load_dataset

from logos import PROVIDER_LOGOS

# ── Design Tokens ─────────────────────────────────────────────────────────────

PROVIDER_COLORS: dict[str, str] = {
    "Google": "#1A73E8",
    "OpenAI": "#10A37F",
    "Meta": "#0866FF",
    "Anthropic": "#D97706",
    "Claude": "#D97706",
    "Grok": "#1D9BF0",
    "Qwen": "#6366F1",
    "Mistral": "#FF6B35",
    "NetoAI": "#06B6D4",
    "IBM": "#0F62FE",
    "IBM Granite": "#0F62FE",
    "DeepSeek": "#8B5CF6",
    "LiquidAI": "#F59E0B",
    "Microsoft": "#00BCF2",
    "Swiss AI": "#EF4444",
    "ByteDance": "#22C55E",
    "Amazon": "#FF9900",
    "NVIDIA": "#76B900",
    "Cohere": "#EF4444",
    "Hugging Face": "#FFB800",
    "Moonshot AI": "#4A90D9",
    "Xiaomi": "#FF6900",
    "xAI": "#1D9BF0",
    "AT&T": "#009FDB",
    "China Telecom": "#E60012",
    "InternLM": "#059669",
    "TII": "#8E5B2C",
    "TSLAM": "#E85D04",
    "SoftBank": "#C0C0C0",
    "MiniMax": "#7C3AED",
}

# ── Benchmark Metadata ────────────────────────────────────────────────────────
# Column order matches the website's BENCHMARKS array (excluding tci and comingSoon)

BENCHMARK_COLS: list[str] = [
    "three_gpp",      # 3GPP-TSG
    "oranbench",      # ORANBench
    "srsranbench",    # srsRANBench
    "telelogs",       # TeleLogs
    "telemath",       # TeleMath
    "teleqna",        # TeleQnA
    "teletables",     # TeleTables
]

BENCHMARK_DISPLAY_NAMES: dict[str, str] = {
    "average": "AVG",
    "teleqna": "TeleQnA",
    "oranbench": "ORANBench",
    "srsranbench": "srsRANBench",
    "telemath": "TeleMath",
    "telelogs": "TeleLogs",
    "three_gpp": "3GPP-TSG",
    "teletables": "TeleTables",
}


# ── Benchmark Methodology Metadata ───────────────────────────────────────────
# Ported from website/src/constants/benchmarks.ts — descriptions, sections,
# artifact links shown in the two-column detail view.

BENCHMARK_METADATA: dict[str, dict] = {
    "teleqna": {
        "title": "TeleQnA",
        "samples": "10,000",
        "category": "Knowledge",
        "description": (
            "TeleQnA is the first benchmark built to measure how well language models "
            "understand telecommunications. It contains 10,000 multiple-choice questions "
            "spanning terminology, research literature, and technical standards from bodies "
            "like 3GPP and IEEE. Questions range from basic definitions to detailed "
            "specification lookups \u2014 the same breadth a working telecom engineer encounters."
        ),
        "sections": [
            {
                "heading": "Methodology",
                "content": (
                    "Questions were generated through an automated framework that extracts "
                    "content from standards documents and research publications, then formulates "
                    "question-answer pairs with distractors. Human reviewers validated quality at "
                    "multiple stages \u2014 filtering ambiguous phrasing, verifying correct answers, "
                    "and ensuring coverage across difficulty levels."
                ),
            },
            {
                "heading": "Dataset",
                "content": (
                    "The 10,000 questions split across five categories: Lexicon (500 questions on "
                    "core terminology), Research Overview (2,000 on broad research topics), Research "
                    "Publications (4,500 from journals and conference proceedings), Standards Overview "
                    "(1,000 on high-level 3GPP/IEEE summaries), and Standards Specifications (2,000 "
                    "on detailed technical implementations)."
                ),
            },
            {
                "heading": "Key Findings",
                "content": (
                    "In the original evaluation, GPT-4 scored 74.9% overall, outperforming active "
                    "telecom professionals (64.9%). But performance drops sharply on Standards "
                    "Specifications \u2014 GPT-4 manages 64.8% where humans score 56.3%. Adding domain "
                    "context via retrieval-augmented generation significantly improved results, "
                    "suggesting that raw parametric knowledge alone isn\u2019t enough for the hardest "
                    "standards questions."
                ),
            },
        ],
        "paperLink": "https://arxiv.org/abs/2310.15051",
        "datasetLink": "https://huggingface.co/datasets/netop/TeleQnA",
    },
    "oranbench": {
        "title": "ORANBench",
        "samples": "4,500",
        "category": "Network Ops",
        "description": (
            "ORANBench is a streamlined evaluation derived from ORAN-Bench-13K, the first "
            "comprehensive benchmark for O-RAN knowledge. It contains 1,500 multiple-choice "
            "questions drawn from 116 O-RAN Alliance specification documents via stratified "
            "sampling across three difficulty tiers (Easy, Medium, Hard). Topics span O-RAN "
            "architecture, open interfaces, the RAN Intelligent Controller (RIC), and working "
            "groups WG1\u2013WG9. The leaderboard evaluates each model over 3 epochs for a total "
            "of 4,500 scored samples."
        ),
        "sections": [
            {
                "heading": "Methodology",
                "content": (
                    "Questions were automatically extracted from O-RAN specification PDFs using "
                    "the ORANSight RAG pipeline, which applies semantic chunking and dense-vector "
                    "retrieval over 2.53 million words of specification text. The full "
                    "ORAN-Bench-13K corpus of 13,952 questions was then reduced to 1,500 through "
                    "stratified sampling \u2014 500 Easy, 500 Medium, and 500 Hard \u2014 preserving "
                    "difficulty balance while keeping evaluation cost manageable. Each question "
                    "is a 4-option multiple-choice item with one correct answer."
                ),
            },
            {
                "heading": "Dataset",
                "content": (
                    "The 1,500 questions are split equally across three difficulty tiers: Easy "
                    "(basic terminology and definitions), Medium (interface behavior and protocol "
                    "interactions), and Hard (cross-specification reasoning requiring synthesis "
                    "of multiple O-RAN documents). Topic coverage includes fronthaul/midhaul "
                    "interfaces, near-RT and non-RT RIC, E2/A1/O1 protocols, network slicing, "
                    "CPRI/eCPRI, NETCONF/YANG, and AI/ML integration for RAN optimization."
                ),
            },
            {
                "heading": "Key Findings",
                "content": (
                    "In the original evaluation, the ORANSight RAG-augmented pipeline achieved "
                    "78.4% macro accuracy, outperforming general-purpose models like GPT-4o and "
                    "Gemini by approximately 5.4%. The performance gap widened on Hard-tier "
                    "questions, where domain-specific retrieval provided the largest advantage. "
                    "This demonstrates that telecom-specialized fine-tuning and retrieval "
                    "significantly improve O-RAN specification comprehension over general "
                    "parametric knowledge alone."
                ),
            },
        ],
        "paperLink": "https://arxiv.org/abs/2407.06245",
        "datasetLink": "https://huggingface.co/datasets/prnshv/ORANBench",
    },
    "srsranbench": {
        "title": "srsRANBench",
        "samples": "4,506",
        "category": "Network Ops",
        "description": (
            "srsRANBench is a code-centric benchmark that evaluates LLM understanding of the "
            "srsRAN 5G O-RAN software stack. It contains 1,502 multiple-choice questions "
            "generated from the srsRAN C++ codebase, testing both code generation and code "
            "comprehension capabilities. The leaderboard evaluates each model over 3 epochs "
            "for a total of 4,506 scored samples."
        ),
        "sections": [
            {
                "heading": "Methodology",
                "content": (
                    "Questions were constructed via the RANSTRUCT dual-agent pipeline, which "
                    "processes 4.68 million words of srsRAN C++ source code using semantic "
                    "chunking and dense-vector encoding. C++ files were randomly selected across "
                    "the srsRAN codebase and converted into 4-option multiple-choice questions "
                    "that test code generation (writing correct implementations) and code "
                    "understanding (reasoning about existing logic, control flow, and API usage)."
                ),
            },
            {
                "heading": "Dataset",
                "content": (
                    "The 1,502 questions cover the major components of the srsRAN stack: the "
                    "Distributed Unit (DU), physical-layer processing (PDCCH, PUCCH, SSS, "
                    "CSI-RS), protocol stacks (F1AP, E1AP, NGAP, SDAP, PDCP), channel "
                    "estimation, CU-UP/CU-CP gateways, and low-level memory management and "
                    "threading primitives. This breadth tests whether models can reason about "
                    "real 5G RAN codebases rather than textbook descriptions."
                ),
            },
            {
                "heading": "Key Findings",
                "content": (
                    "ORANSight-2.0 fine-tuned models outperformed GPT-4o and Gemini by 18.5% "
                    "on srsRANBench \u2014 the largest gap of any benchmark in the suite. This "
                    "suggests that general-purpose LLMs lack familiarity with telecom-specific "
                    "codebases and that domain-specific fine-tuning yields even larger gains on "
                    "code understanding tasks than on specification knowledge tasks. The result "
                    "highlights a significant opportunity for specialized models in RAN software "
                    "engineering."
                ),
            },
        ],
        "paperLink": "https://arxiv.org/abs/2503.05200",
        "datasetLink": "https://huggingface.co/datasets/prnshv/srsRANBench",
    },
    "telemath": {
        "title": "TeleMath",
        "samples": "500",
        "category": "Knowledge",
        "description": (
            "TeleMath is the first benchmark designed to evaluate LLM performance on mathematical "
            "problem-solving in telecommunications. It contains 500 question-answer pairs with "
            "strictly numerical solutions \u2014 signal-to-noise ratios, throughput calculations, "
            "queueing theory results \u2014 the kind of quantitative reasoning a telecom engineer "
            "does daily."
        ),
        "sections": [
            {
                "heading": "Methodology",
                "content": (
                    "Subject matter experts authored 100 seed problems with detailed, step-by-step "
                    "solutions. An automated pipeline then decomposed each solution into standalone "
                    "subproblems and built reusable blueprints via two parallel paths: a code-driven "
                    "path (converting numerical steps into executable Python validated against "
                    "expected answers) and a symbolic math path (standardizing equations to LaTeX "
                    "and extracting parameterized templates via SymPy)."
                ),
            },
            {
                "heading": "Dataset",
                "content": (
                    "The 500 problems span seven categories: Telecommunications Engineering (30.6%), "
                    "Electrical Engineering, Signal Processing, Information Theory, Computer "
                    "Networking, Probability & Statistics, and Operations Research. All answers are "
                    "strictly numerical (float or integer), ensuring unambiguous evaluation."
                ),
            },
            {
                "heading": "Key Findings",
                "content": (
                    "Models built for reasoning dominate. Qwen3-32B leads with 76.0% accuracy "
                    "(cons@16), while general-purpose models struggle \u2014 Llama-3.3-70B, despite being "
                    "twice the size, scores only 40.2%. Math-specialized models also underperformed "
                    "reasoning models, suggesting that generic mathematical training alone isn\u2019t "
                    "enough for domain-specific technical problems."
                ),
            },
        ],
        "paperLink": "https://arxiv.org/abs/2506.10674",
        "datasetLink": "https://huggingface.co/datasets/netop/TeleMath",
    },
    "telelogs": {
        "title": "TeleLogs",
        "samples": "1,000+",
        "category": "Network Ops",
        "description": (
            "TeleLogs evaluates how well language models perform root cause analysis on 5G "
            "network issues. Each problem presents realistic drive test telemetry \u2014 signal "
            "strength, throughput, interference metrics, handover events \u2014 alongside engineering "
            "parameters, and asks the model to diagnose why downlink throughput degraded below "
            "600 Mbps."
        ),
        "sections": [
            {
                "heading": "Methodology",
                "content": (
                    "The dataset is synthetic but seeded from realistic network engineering "
                    "parameters. Each scenario simulates a UE drive test through a region covered "
                    "by multiple 5G gNodeBs, with user-plane measurements (RSRP, SINR, throughput, "
                    "speed, PCI) and base station configurations. A two-stage fine-tuning approach "
                    "\u2014 supervised fine-tuning with chain-of-thought traces followed by "
                    "reinforcement learning (GRPO) \u2014 was used to produce structured, multi-step "
                    "diagnostic reasoning."
                ),
            },
            {
                "heading": "Dataset",
                "content": (
                    "The dataset contains over 2,800 annotated troubleshooting problems split into "
                    "training and test sets. Problems map to eight specific root causes: excessive "
                    "vehicle speed, downtilt angle too large, serving cell overshooting beyond 1 km, "
                    "co-frequency neighbor interference, PCI mod 30 collision, frequent handovers, "
                    "misconfigured handover thresholds, and insufficient scheduled resource blocks."
                ),
            },
            {
                "heading": "Key Findings",
                "content": (
                    "Off-the-shelf reasoning models struggle with these problems, exposing a gap in "
                    "domain-specific network diagnostics. Domain-adapted models dramatically close "
                    "this gap \u2014 a fine-tuned Qwen2.5-32B achieves 95.86% accuracy (pass@1) on the "
                    "test set and 93.23% on randomized variants."
                ),
            },
        ],
        "paperLink": "https://arxiv.org/abs/2507.21974",
        "datasetLink": "https://huggingface.co/datasets/netop/TeleLogs",
    },
    "three_gpp": {
        "title": "3GPP-TSG",
        "samples": "5,000+",
        "category": "Knowledge",
        "description": (
            "3GPP-TSG tests whether language models can identify which 3GPP working group "
            "authored a given technical document passage. It requires models to classify text "
            "excerpts from 3GPP specifications into the correct working group \u2014 a task that "
            "demands deep familiarity with the scope, terminology, and technical focus of each "
            "group."
        ),
        "sections": [
            {
                "heading": "Methodology",
                "content": (
                    "Text passages are extracted directly from 3GPP technical documents (Tdocs) "
                    "and presented to models with no additional context. Models must classify each "
                    "passage into one of 16 working groups in a zero-shot setting. The task is pure "
                    "text classification \u2014 no multiple-choice hints or narrowed option sets."
                ),
            },
            {
                "heading": "Dataset",
                "content": (
                    "The benchmark spans 16 working groups across three Technical Specification "
                    "Groups: Radio Access Network (RAN1\u20135), System Architecture (SA1\u20136), and Core "
                    "Network & Terminals (CT1, CT3, CT4, CT6). Passages are drawn from 3GPP "
                    "standards covering releases 8 through 19."
                ),
            },
            {
                "heading": "Key Findings",
                "content": (
                    "General-purpose models perform poorly \u2014 GPT-4o achieves only 38.9% accuracy, "
                    "barely above random for 16 classes. Telecom-adapted models dramatically "
                    "outperform: TelecomGPT reaches 75.3%, with particularly strong performance on "
                    "RAN documents (82.8%). The CT group proves hardest, with even fine-tuned models "
                    "scoring below 50% on some configurations."
                ),
            },
        ],
        "paperLink": "https://arxiv.org/abs/2407.09424",
        "datasetLink": "https://huggingface.co/datasets/eaguaida/gsma_sample",
    },
    "teletables": {
        "title": "TeleTables",
        "samples": "500",
        "category": "Knowledge",
        "description": (
            "TeleTables evaluates how well language models understand and reason over tables "
            "embedded in 3GPP technical specifications. It contains 500 multiple-choice questions "
            "drawn from 2,220 tables across 13 3GPP documents (Release 18 and 19). The benchmark "
            "tests two distinct capabilities: implicit knowledge of standards content and explicit "
            "interpretation of tabular data."
        ),
        "sections": [
            {
                "heading": "Methodology",
                "content": (
                    "A multi-stage pipeline generated questions from tables extracted from 3GPP "
                    "specs. First, a multimodal generator created basic MCQs from tables in four "
                    "formats \u2014 HTML, JSON, Markdown, and PNG image \u2014 with a validator retaining "
                    "only questions answered correctly in at least 3 of 4 independent trials. A "
                    "reasoning model then synthesized sets of basic questions into harder variants "
                    "requiring multi-step reasoning."
                ),
            },
            {
                "heading": "Dataset",
                "content": (
                    "The 500 MCQs are balanced across easy and difficult tiers. Each question "
                    "comes with five candidate answers, a correct answer, an explanation, and a "
                    "difficulty flag. Tables are provided in four representations: HTML (avg. 1,224 "
                    "tokens), JSON (893 tokens), Markdown (707 tokens), and PNG images."
                ),
            },
            {
                "heading": "Key Findings",
                "content": (
                    "Without table context, models average just ~35% accuracy \u2014 they simply "
                    "haven\u2019t memorized 3GPP table content during pretraining. With tables provided, "
                    "reasoning models close the gap dramatically: Qwen3-32B reaches 92.6% overall. "
                    "HTML is the optimal format for most models, though Markdown (42% fewer tokens) "
                    "offers a practical efficiency trade-off."
                ),
            },
        ],
        "paperLink": "https://arxiv.org/abs/2601.04202",
        "datasetLink": "https://huggingface.co/datasets/netop/TeleTables",
    },
}

# ── Data Loading ──────────────────────────────────────────────────────────────


def parse_score_string(value) -> tuple[float | None, float | None]:
    """Parse score strings like '[0.852, 0.0112]' from the HF dataset."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, None
    try:
        if isinstance(value, str):
            parsed = ast.literal_eval(value)
        else:
            parsed = value
        if not isinstance(parsed, (list, tuple)) or len(parsed) < 2:
            return None, None
        score = float(parsed[0]) * 100
        stderr = float(parsed[1]) * 100
        return score, stderr
    except (ValueError, SyntaxError, TypeError):
        return None, None


def get_provider_color(provider: str) -> str:
    if provider in PROVIDER_COLORS:
        return PROVIDER_COLORS[provider]
    lower = provider.lower()
    for key, color in PROVIDER_COLORS.items():
        if key.lower() == lower:
            return color
    return "#6B7280"


def get_provider_logo(provider: str) -> str:
    """Return base64 data URI for provider logo, or empty string."""
    if provider in PROVIDER_LOGOS:
        return PROVIDER_LOGOS[provider]
    lower = provider.lower()
    for key, uri in PROVIDER_LOGOS.items():
        if key.lower() == lower:
            return uri
    return ""


def load_leaderboard_data() -> pd.DataFrame:
    """Load GSMA/leaderboard HF dataset and prepare a flat DataFrame."""
    ds = load_dataset("GSMA/leaderboard", split="train", download_mode="force_redownload")
    df = ds.to_pandas()

    if "__index_level_0__" in df.columns:
        df = df.drop(columns=["__index_level_0__"])

    df["model_name"] = df["model"]
    df["avg_pct"] = df["average"].apply(
        lambda x: round(x * 100, 1) if pd.notna(x) else 0
    )

    for col in BENCHMARK_COLS:
        if col in df.columns:
            scores = df[col].apply(parse_score_string)
            df[f"{col}_score"] = scores.apply(lambda x: x[0]).fillna(0)
            df[f"{col}_stderr"] = scores.apply(lambda x: x[1]).fillna(0)

    df = df.sort_values("rank", ascending=True, na_position="last").reset_index(
        drop=True
    )
    return df


# ── Cached Accessor ───────────────────────────────────────────────────────────
# The Space process is long-lived, so loading once at startup would freeze the
# table at boot-time data. Re-read the dataset on a TTL so the leaderboard stays
# in sync with GSMA/leaderboard without a restart. The TTL prevents a redownload
# on every interaction; an 87-row dataset makes the periodic refetch negligible.

_DATA_TTL_SECONDS = 600  # re-read at most every 10 minutes
_data_cache: dict = {"df": None, "ts": 0.0}


def get_leaderboard_data() -> pd.DataFrame:
    """Return leaderboard data, re-reading the dataset when the cache is stale."""
    now = time.time()
    if _data_cache["df"] is None or (now - _data_cache["ts"]) > _DATA_TTL_SECONDS:
        _data_cache["df"] = load_leaderboard_data()
        _data_cache["ts"] = now
    return _data_cache["df"]


# ── HTML Rendering ────────────────────────────────────────────────────────────


def render_hero() -> str:
    """Render the hero section with image and title."""
    hero_path = Path(__file__).parent / "leaderboard.png"
    if hero_path.exists():
        with open(hero_path, "rb") as f:
            hero_b64 = base64.b64encode(f.read()).decode()
        img_src = f"data:image/png;base64,{hero_b64}"
    else:
        img_src = ""

    img_html = (
        f'<img class="ot-hero-image" src="{img_src}" alt="Leaderboard">'
        if img_src
        else ""
    )

    return f"""
    <div class="ot-hero">
        {img_html}
        <div class="ot-hero-content">
            <h1 class="ot-hero-title">Open Telco AI Leaderboard</h1>
        </div>
    </div>
    """


def render_sidebar(active_key: str = "average") -> str:
    """Render the sidebar navigation as pure HTML with JS interactions."""
    lb_active = " active" if active_key == "average" else ""
    html = f"""
    <button class="sidebar-lb-btn{lb_active}" onclick="selectBenchmark('average')">Leaderboard</button>
    <div class="sidebar-divider"></div>
    <div class="sidebar-heading">Benchmarks</div>
    """
    for bm in BENCHMARK_COLS:
        display = BENCHMARK_DISPLAY_NAMES.get(bm, bm)
        active_cls = " active" if active_key == bm else ""
        html += (
            f'<button class="sidebar-item{active_cls}" '
            f"""onclick="selectBenchmark('{bm}')">{display}</button>"""
        )
    return html


def render_rank(rank: int) -> str:
    """Render rank as plain number with color for top 3."""
    if rank == 1:
        cls = "rank-num rank-1"
    elif rank == 2:
        cls = "rank-num rank-2"
    elif rank == 3:
        cls = "rank-num rank-3"
    else:
        cls = "rank-num"
    return f'<span class="{cls}">{rank}</span>'


def render_model_cell(name: str, provider: str) -> str:
    """Render model cell with provider logo and name."""
    safe_name = html_mod.escape(str(name))
    logo_uri = get_provider_logo(provider)
    if logo_uri:
        logo_html = f'<img class="model-logo" src="{logo_uri}" alt="">'
    else:
        color = get_provider_color(provider)
        logo_html = (
            f'<span class="model-logo" style="display:inline-block;'
            f"background:{color};border-radius:8px;width:18px;height:18px;"
            f'flex-shrink:0"></span>'
        )
    return f'<div class="model-cell">{logo_html}<span class="model-name">{safe_name}</span></div>'


def render_avg_cell(avg_pct: float | None) -> str:
    """Render average score as plain number (no %, no badge)."""
    if avg_pct is None or (isinstance(avg_pct, float) and pd.isna(avg_pct)):
        return '<span class="avg-score">0.0</span>'
    return f'<span class="avg-score">{avg_pct:.1f}</span>'


def render_score_cell(score: float | None, stderr: float | None) -> str:
    if score is None or (isinstance(score, float) and pd.isna(score)):
        return '<span class="score-main">0.0</span>'
    stderr_val = stderr if stderr is not None and not (isinstance(stderr, float) and pd.isna(stderr)) else 0
    # Absolute stderr — matches what the visual error bars show on the chart
    if stderr_val and stderr_val > 0:
        stderr_html = f'<span class="score-stderr">&plusmn;{stderr_val:.1f}</span>'
    else:
        stderr_html = ""
    return f'<span class="score-main">{score:.1f}</span>{stderr_html}'


def render_leaderboard_table(
    df: pd.DataFrame,
    sort_by: str = "average",
    ascending: bool = False,
) -> str:
    """Render the full leaderboard as an HTML table with clickable headers."""
    # Sort
    if sort_by == "average":
        sort_col = "avg_pct"
    else:
        sort_col = f"{sort_by}_score"

    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=ascending, na_position="last")
        df = df.reset_index(drop=True)

    arrow = "↑" if ascending else "↓"
    dir_str = "asc" if ascending else "desc"

    # Header row — pass current sort column so JS can toggle vs default
    avg_active = " col-active" if sort_by == "average" else ""
    avg_arrow = f'<span class="sort-arrow">{arrow}</span>' if sort_by == "average" else ""
    header_cells = [
        '<th class="col-rank">#</th>',
        '<th class="col-model">Model</th>',
        f'<th class="col-avg{avg_active}" onclick="sortTable(\'average\', \'{sort_by}\', \'{dir_str}\')">AVG{avg_arrow}</th>',
    ]

    for col in BENCHMARK_COLS:
        display = BENCHMARK_DISPLAY_NAMES.get(col, col)
        active = " col-active" if sort_by == col else ""
        col_arrow = f'<span class="sort-arrow">{arrow}</span>' if sort_by == col else ""
        header_cells.append(
            f"""<th class="col-score{active}" onclick="sortTable('{col}', '{sort_by}', '{dir_str}')">{display}{col_arrow}</th>"""
        )

    rows_html = []
    for i, (_, row) in enumerate(df.iterrows()):
        rank = i + 1
        model_html = render_model_cell(row["model_name"], row["provider"])
        avg_html = render_avg_cell(row.get("avg_pct"))

        cells = [
            f'<td class="col-rank">{render_rank(rank)}</td>',
            f'<td class="col-model">{model_html}</td>',
            f'<td class="col-avg{avg_active}">{avg_html}</td>',
        ]

        for col in BENCHMARK_COLS:
            score = row.get(f"{col}_score")
            stderr = row.get(f"{col}_stderr")
            active = " col-active" if sort_by == col else ""
            cells.append(
                f'<td class="col-score{active}">{render_score_cell(score, stderr)}</td>'
            )

        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    return f"""
    <div class="table-wrapper">
        <table class="leaderboard-table">
            <thead><tr>{''.join(header_cells)}</tr></thead>
            <tbody>{''.join(rows_html)}</tbody>
        </table>
    </div>
    """


def render_benchmark_bars(df: pd.DataFrame, benchmark: str) -> str:
    """Render horizontal bar chart for a single benchmark."""
    score_col = f"{benchmark}_score"
    stderr_col = f"{benchmark}_stderr"
    display = BENCHMARK_DISPLAY_NAMES.get(benchmark, benchmark)

    if score_col not in df.columns:
        return f'<div class="bar-empty">No data for {display}</div>'

    sorted_df = (
        df.dropna(subset=[score_col])
        .sort_values(score_col, ascending=False)
        .reset_index(drop=True)
    )

    if sorted_df.empty:
        return f'<div class="bar-empty">No data for {display}</div>'

    max_with_error = 100  # All benchmarks are 0-100 percentage scores

    bars_html = []
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        rank = i + 1
        score = row[score_col]
        stderr = row.get(stderr_col)
        pct = (score / max_with_error) * 100
        provider_color = get_provider_color(row["provider"])
        safe_name = html_mod.escape(str(row["model_name"]))

        # Logo
        logo_uri = get_provider_logo(row["provider"])
        if logo_uri:
            logo_html = f'<img class="model-logo" src="{logo_uri}" alt="">'
        else:
            logo_html = (
                f'<span class="model-logo" style="display:inline-block;'
                f"background:{provider_color};border-radius:8px;width:18px;height:18px;"
                f'flex-shrink:0"></span>'
            )

        # Absolute stderr — matches visual error bar width on the chart
        if stderr and pd.notna(stderr) and stderr > 0:
            stderr_html = f'<span class="score-stderr">&plusmn;{stderr:.1f}</span>'
        else:
            stderr_html = ""

        # Rank badge class
        rank_cls = "bar-rank-badge"
        if rank <= 3:
            rank_cls += f" rank-{rank}"

        # CI ticks
        ci_html = ""
        if stderr and pd.notna(stderr) and stderr > 0:
            ci_low_pct = max(0, ((score - stderr) / max_with_error) * 100)
            ci_high_pct = min(100, ((score + stderr) / max_with_error) * 100)
            ci_html = (
                f'<div class="ci-tick" style="left:{ci_low_pct:.2f}%"></div>'
                f'<div class="ci-tick" style="left:{ci_high_pct:.2f}%"></div>'
                f'<div class="ci-connector" style="left:{ci_low_pct:.2f}%;'
                f'width:{ci_high_pct - ci_low_pct:.2f}%"></div>'
            )

        delay = i * 0.04 + 0.02

        bars_html.append(f"""
        <div class="bar-entry">
            <div class="{rank_cls}">{rank}</div>
            <div class="bar-entry-content">
                <div class="bar-header">
                    <div class="bar-model-info">
                        {logo_html}
                        <span class="bar-model-name">{safe_name}</span>
                    </div>
                    <div class="bar-score-label">
                        <span class="score-main">{score:.1f}</span>{stderr_html}
                    </div>
                </div>
                <div class="bar-track-wrapper">
                    <div class="bar-track">
                        <div class="bar-fill" style="width:{pct:.1f}%;background:{provider_color};animation-delay:{delay:.2f}s"></div>
                    </div>
                    {ci_html}
                </div>
            </div>
        </div>
        """)

    return f"""
    <div class="bar-chart">
        <h2 class="bar-title">{display}</h2>
        <div class="bar-list">{''.join(bars_html)}</div>
    </div>
    """


def render_benchmark_methodology(benchmark: str) -> str:
    """Render HTML for the left prose column of the benchmark detail view."""
    meta = BENCHMARK_METADATA.get(benchmark)
    if not meta:
        display = BENCHMARK_DISPLAY_NAMES.get(benchmark, benchmark)
        return f'<h1 class="bm-title">{html_mod.escape(display)}</h1>'

    title = html_mod.escape(meta["title"])
    desc = html_mod.escape(meta["description"])

    # Metric pills
    pills_html = ""
    if meta.get("samples"):
        pills_html += f'<span class="bm-pill">Samples: {html_mod.escape(meta["samples"])}</span>'
    if meta.get("category"):
        pills_html += f'<span class="bm-pill">{html_mod.escape(meta["category"])}</span>'

    # Sections
    sections_html = ""
    for section in meta.get("sections", []):
        heading = html_mod.escape(section["heading"])
        content = html_mod.escape(section["content"])
        sections_html += f'<h3 class="bm-section-heading">{heading}</h3>'
        sections_html += f'<p class="bm-section-content">{content}</p>'

    # Artifact links
    artifacts_html = ""
    paper = meta.get("paperLink")
    dataset = meta.get("datasetLink")
    if paper or dataset:
        artifacts_html += '<h3 class="bm-section-heading">Release Artifacts</h3><div class="bm-artifacts">'
        if paper:
            safe_paper = html_mod.escape(paper)
            artifacts_html += (
                f'<a href="{safe_paper}" target="_blank" rel="noopener noreferrer" class="bm-artifact-link">'
                '<svg class="bm-artifact-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">'
                '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
                '<polyline points="14 2 14 8 20 8"/>'
                '<line x1="16" y1="13" x2="8" y2="13"/>'
                '<line x1="16" y1="17" x2="8" y2="17"/>'
                '<polyline points="10 9 9 9 8 9"/>'
                "</svg>Paper (arXiv)</a>"
            )
        if dataset:
            safe_dataset = html_mod.escape(dataset)
            artifacts_html += (
                f'<a href="{safe_dataset}" target="_blank" rel="noopener noreferrer" class="bm-artifact-link">'
                '<svg class="bm-artifact-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">'
                '<ellipse cx="12" cy="5" rx="9" ry="3"/>'
                '<path d="M21 12c0 1.66-4.03 3-9 3s-9-1.34-9-3"/>'
                '<path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/>'
                "</svg>Dataset (HuggingFace)</a>"
            )
        artifacts_html += "</div>"

    return f"""
    <h1 class="bm-title">{title}</h1>
    <div class="bm-metrics">{pills_html}</div>
    <p class="bm-description">{desc}</p>
    {sections_html}
    <hr class="bm-separator">
    {artifacts_html}
    """


def render_footer() -> str:
    return """
    <div class="ot-footer">
        <p>
            Open Telco Leaderboard &middot;
            <a href="https://github.com/gsma-labs/leaderboard" target="_blank">GitHub</a> &middot;
            <a href="https://huggingface.co/datasets/GSMA/leaderboard" target="_blank">Dataset</a> &middot;
            GSMA Research
        </p>
    </div>
    """


# ── Gradio App ────────────────────────────────────────────────────────────────

CSS_PATH = Path(__file__).parent / "style.css"

_VALID_KEYS = {"average", *BENCHMARK_COLS}


def _strip_timestamp(value: str) -> str:
    """Strip JS timestamp suffix (e.g. 'teleqna_1234567890' → 'teleqna').

    Handles keys with underscores like 'three_gpp' by checking against known keys.
    """
    if value in _VALID_KEYS:
        return value
    # Try stripping last _DIGITS segment
    parts = value.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and parts[0] in _VALID_KEYS:
        return parts[0]
    return value


def switch_view(nav_value: str) -> tuple[str, str, str, str, str]:
    """Return (hero, main_table, prose, chart, sidebar) for the selected view."""
    benchmark = _strip_timestamp(nav_value)
    df = get_leaderboard_data()
    sidebar_html = render_sidebar(active_key=benchmark)
    if benchmark == "average":
        return (
            render_hero(),
            render_leaderboard_table(df, sort_by="average", ascending=False),
            "",
            "",
            sidebar_html,
        )
    else:
        return (
            "",
            "",
            render_benchmark_methodology(benchmark),
            render_benchmark_bars(df, benchmark),
            sidebar_html,
        )


def sort_table_handler(payload: str) -> tuple[str, str, str, str, str]:
    """Sort table by column. Payload format: 'col|dir|timestamp'."""
    parts = payload.split("|")
    sort_col = parts[0] if len(parts) >= 1 else "average"
    direction = parts[1] if len(parts) >= 2 else "desc"
    if sort_col not in _VALID_KEYS:
        sort_col = "average"
    ascending = direction == "asc"
    df = get_leaderboard_data()
    return (
        render_hero(),
        render_leaderboard_table(df, sort_by=sort_col, ascending=ascending),
        "",
        "",
        render_sidebar(active_key="average"),
    )


def refresh_overview() -> tuple[str, str, str]:
    """Render the default overview with current data (runs on each page load)."""
    df = get_leaderboard_data()
    return (
        render_hero(),
        render_leaderboard_table(df, sort_by="average", ascending=False),
        render_sidebar(active_key="average"),
    )


def build_app() -> gr.Blocks:
    css_text = CSS_PATH.read_text() if CSS_PATH.exists() else ""
    with gr.Blocks(
        title="Open Telco Leaderboard",
        css=css_text,
        head=JS_CODE,
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="zinc",
            font=gr.themes.GoogleFont("Inter"),
        ),
    ) as app:
        # Hero — dynamic output (hidden when viewing benchmark detail)
        hero_display = gr.HTML(render_hero(), elem_classes=["ot-hero-wrap"], apply_default_css=False)

        with gr.Row(elem_classes=["ot-main"]):
            # ── Sidebar ──
            with gr.Column(scale=0, min_width=200, elem_classes=["ot-sidebar"]):
                sidebar_html = gr.HTML(
                    render_sidebar(active_key="average"),
                    elem_classes=["sidebar-inner"],
                    apply_default_css=False,
                )

            # ── Content ──
            with gr.Column(scale=4, elem_classes=["ot-content"]):
                # Region 1: Leaderboard table (shown for overall view)
                main_display = gr.HTML(
                    render_leaderboard_table(get_leaderboard_data()),
                    elem_classes=["ot-display"],
                    apply_default_css=False,
                )
                # Region 2: Two-column detail (shown for benchmark view)
                with gr.Row(elem_classes=["bm-detail-row"], visible=True):
                    with gr.Column(scale=1, min_width=300, elem_classes=["bm-prose-wrap"]):
                        prose_display = gr.HTML("", elem_classes=["bm-prose-inner"], apply_default_css=False)
                    with gr.Column(scale=1, min_width=300, elem_classes=["bm-chart-wrap"]):
                        chart_display = gr.HTML("", elem_classes=["bm-chart-inner"], apply_default_css=False)

        # Footer
        gr.HTML(render_footer(), elem_classes=["ot-footer-wrap"], apply_default_css=False)

        # ── JS→Python bridge using Gradio buttons (reliable across versions) ──
        # Hidden via CSS (not visible=False, which removes from DOM entirely)
        with gr.Row(visible=True, elem_id="hidden-controls"):
            nav_input = gr.Textbox(elem_id="nav-input")
            sort_input = gr.Textbox(elem_id="sort-col-input")

        nav_input.change(
            fn=switch_view,
            inputs=[nav_input],
            outputs=[hero_display, main_display, prose_display, chart_display, sidebar_html],
        )
        sort_input.change(
            fn=sort_table_handler,
            inputs=[sort_input],
            outputs=[hero_display, main_display, prose_display, chart_display, sidebar_html],
        )

        # Refresh the table on every page load so new visitors see current data;
        # the TTL cache keeps this from re-downloading the dataset on each visit.
        app.load(
            fn=refresh_overview,
            inputs=None,
            outputs=[hero_display, main_display, sidebar_html],
        )

    return app


# JavaScript injected into the page for sidebar and sort interactions
JS_CODE = """
<script>
function _setGradioValue(elemId, value) {
    const wrapper = document.getElementById(elemId);
    console.log('[OT] _setGradioValue', elemId, '→ wrapper:', wrapper);
    if (!wrapper) {
        console.error('[OT] Element #' + elemId + ' NOT FOUND. DOM ids:',
            Array.from(document.querySelectorAll('[id]')).map(e => e.id).filter(id => id.includes('nav') || id.includes('sort') || id.includes('hidden')));
        return;
    }
    // Try input first (Gradio 5+/6 single-line), then textarea (older/multi-line)
    const el = wrapper.querySelector('input') || wrapper.querySelector('textarea');
    console.log('[OT] Found inner element:', el?.tagName, el);
    if (!el) {
        console.error('[OT] No input/textarea inside #' + elemId + '. innerHTML:', wrapper.innerHTML.substring(0, 200));
        return;
    }
    const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
    setter.call(el, value);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    console.log('[OT] Value set to:', value);
}

function selectBenchmark(key) {
    console.log('[OT] selectBenchmark called:', key);
    _setGradioValue('nav-input', key + '_' + Date.now());
    // Toggle visibility: table vs detail row
    _toggleDetailView(key !== 'average');
}

function _toggleDetailView(showDetail) {
    // Find wrappers by class; wait a tick for Gradio to update DOM
    setTimeout(function() {
        var display = document.querySelector('.ot-display');
        var detailRow = document.querySelector('.bm-detail-row');
        if (display) display.style.display = showDetail ? 'none' : '';
        if (detailRow) detailRow.style.display = showDetail ? '' : 'none';
    }, 50);
}

function sortTable(col, currentSortCol, currentDir) {
    const newDir = (col === currentSortCol)
        ? (currentDir === 'desc' ? 'asc' : 'desc')
        : 'desc';
    _setGradioValue('sort-col-input', col + '|' + newDir + '|' + Date.now());
    _toggleDetailView(false);
}

// Hide detail row on initial load (leaderboard table shown first)
document.addEventListener('DOMContentLoaded', function() {
    var detailRow = document.querySelector('.bm-detail-row');
    if (detailRow) detailRow.style.display = 'none';
});

</script>
"""


demo = build_app()

if __name__ == "__main__":
    demo.launch()
