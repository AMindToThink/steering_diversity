#\!/usr/bin/env python3
"""Interactive UMAP visualization of steered LLM responses.

Reads responses from a JSONL file, computes (or loads cached) sentence embeddings,
runs UMAP for dimensionality reduction, and produces an interactive Plotly scatter
plot colored by steering scale. Hover tooltips show prompt, response, and metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import umap
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_RESPONSES = "outputs/happy_recon/responses.jsonl"
DEFAULT_EMBEDDINGS = "outputs/happy_recon/embeddings.npz"
DEFAULT_UMAP_COORDS = "outputs/happy_recon/umap_coords.npz"
DEFAULT_OUTPUT_DIR = "outputs/happy_recon/plots"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_responses(path: Path) -> list[dict]:
    """Load JSONL response records."""
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_embeddings(texts: list[str], model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """Compute sentence embeddings using SentenceTransformer."""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"Embedding {len(texts)} texts ...")
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def get_or_compute_embeddings(
    records: list[dict],
    embeddings_path: Path,
) -> np.ndarray:
    """Load cached embeddings if available and aligned, otherwise compute and cache."""
    if embeddings_path.exists():
        print(f"Loading cached embeddings from {embeddings_path}")
        data = np.load(embeddings_path, allow_pickle=True)
        embeddings = data["embeddings"]
        if embeddings.shape[0] == len(records):
            # Verify alignment via scales and prompt_indices if present
            if "scales" in data and "prompt_indices" in data:
                file_scales = data["scales"]
                file_pindices = data["prompt_indices"]
                record_scales = np.array([r["scale"] for r in records], dtype=np.float32)
                record_pindices = np.array([r["prompt_idx"] for r in records], dtype=np.int32)
                if np.allclose(file_scales, record_scales) and np.array_equal(
                    file_pindices, record_pindices
                ):
                    print("Embeddings aligned with responses -- using cached.")
                    return embeddings
                else:
                    print("WARNING: Cached embeddings metadata does not match responses. Recomputing.")
            else:
                print("No metadata in cached embeddings to verify alignment. Using by count match.")
                return embeddings
        else:
            print(
                f"WARNING: Cached embeddings count ({embeddings.shape[0]}) \!= "
                f"responses count ({len(records)}). Recomputing."
            )

    texts = [r["response"] for r in records]
    embeddings = compute_embeddings(texts)

    # Cache for future runs
    scales = np.array([r["scale"] for r in records], dtype=np.float32)
    prompt_indices = np.array([r["prompt_idx"] for r in records], dtype=np.int32)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(embeddings_path, embeddings=embeddings, scales=scales, prompt_indices=prompt_indices)
    print(f"Saved embeddings cache to {embeddings_path}")

    return embeddings


def load_or_compute_umap(
    embeddings: np.ndarray,
    umap_coords_path: Path,
    seed: int = 42,
) -> np.ndarray:
    """Load saved UMAP coordinates if available and aligned, otherwise compute."""
    if umap_coords_path.exists():
        data = np.load(umap_coords_path)
        coords = data["coords"]
        if coords.shape[0] == embeddings.shape[0]:
            print(f"Loaded UMAP coordinates from {umap_coords_path}")
            return coords
        print(
            f"WARNING: Saved UMAP coords count ({coords.shape[0]}) != "
            f"embeddings count ({embeddings.shape[0]}). Recomputing."
        )
    else:
        print(f"No saved UMAP coordinates at {umap_coords_path}, computing from scratch.")

    print(f"Running UMAP on {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]}) ...")
    reducer = umap.UMAP(random_state=seed)
    coords = reducer.fit_transform(embeddings)
    return coords


def truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len characters, adding ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def build_interactive_plot(
    records: list[dict],
    coords: np.ndarray,
) -> tuple[go.Figure, list[int], list[float]]:
    """Build an interactive Plotly scatter plot of the UMAP coordinates.

    One trace per (scale, prompt) combination. Traces sharing a scale get the
    same color and a single legend group. The prompt filter dropdown toggles
    trace *visibility* (not opacity) so hidden points can't capture hover.

    Returns (fig, unique_prompts, unique_scales).
    """
    import plotly.express as px

    unique_scales = sorted(set(r["scale"] for r in records))
    unique_prompts = sorted(set(r["prompt_idx"] for r in records))

    colors = px.colors.qualitative.Vivid
    scale_colors = {s: colors[i % len(colors)] for i, s in enumerate(unique_scales)}

    fig = go.Figure()

    # One trace per (scale, prompt) — traces in same scale share legendgroup
    for scale_val in unique_scales:
        first_in_group = True
        for prompt_idx in unique_prompts:
            idxs = [
                i for i, r in enumerate(records)
                if r["scale"] == scale_val and r["prompt_idx"] == prompt_idx
            ]
            if not idxs:
                continue

            hover_texts = [
                f"prompt={records[i]['prompt_idx']}  scale={records[i]['scale']}  resp={records[i]['response_idx']}"
                for i in idxs
            ]
            customdata = [
                [records[i]["prompt_idx"], records[i]["scale"], records[i]["response_idx"],
                 truncate(records[i]["prompt"].replace("\n", " "), 300),
                 records[i]["response"]]
                for i in idxs
            ]

            fig.add_trace(
                go.Scatter(
                    x=coords[idxs, 0],
                    y=coords[idxs, 1],
                    mode="markers",
                    name=f"scale {scale_val}",
                    legendgroup=f"scale {scale_val}",
                    showlegend=first_in_group,
                    marker=dict(
                        color=scale_colors[scale_val],
                        size=6,
                        opacity=0.8,
                        line=dict(width=0.5, color="white"),
                    ),
                    text=hover_texts,
                    hoverinfo="text",
                    customdata=customdata,
                )
            )
            first_in_group = False

    fig.update_layout(
        title="Interactive UMAP — toggle scales in legend, filter prompts below",
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        width=700,
        height=700,
        template="plotly_white",
        hoverlabel=dict(bgcolor="rgba(255,255,255,0.9)", font_size=12),
        legend=dict(title="Steering Scale", itemclick="toggle", itemdoubleclick="toggleothers",
                    groupclick="togglegroup"),
    )
    return fig, unique_prompts, unique_scales


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive UMAP visualization of steered LLM responses."
    )
    parser.add_argument(
        "--responses",
        type=str,
        default=DEFAULT_RESPONSES,
        help=f"Path to responses JSONL (default: {DEFAULT_RESPONSES})",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default=DEFAULT_EMBEDDINGS,
        help=f"Path to embeddings .npz cache (default: {DEFAULT_EMBEDDINGS})",
    )
    parser.add_argument(
        "--umap-coords",
        type=str,
        default=DEFAULT_UMAP_COORDS,
        help=f"Path to saved UMAP coordinates .npz (default: {DEFAULT_UMAP_COORDS})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for HTML plot (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for UMAP (default: 42)",
    )
    args = parser.parse_args()

    responses_path = Path(args.responses)
    embeddings_path = Path(args.embeddings)
    umap_coords_path = Path(args.umap_coords)
    output_dir = Path(args.output_dir)

    if not responses_path.exists():
        print(f"ERROR: Responses file not found: {responses_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Load responses
    records = load_responses(responses_path)
    print(f"Loaded {len(records)} responses from {responses_path}")

    # 2. Get or compute embeddings
    embeddings = get_or_compute_embeddings(records, embeddings_path)
    assert embeddings.shape[0] == len(records), (
        f"Embedding count ({embeddings.shape[0]}) \!= response count ({len(records)})"
    )

    # 3. Load saved UMAP coordinates or compute from scratch
    coords = load_or_compute_umap(embeddings, umap_coords_path, seed=args.seed)

    # 4. Build and save interactive plot
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "interactive_umap.html"
    fig, unique_prompts, unique_scales = build_interactive_plot(records, coords)
    fig.write_html(str(html_path), include_plotlyjs=True)

    # Build prompt filter options
    prompt_options_html = '<option value="all" selected>All prompts</option>\n'
    for p in unique_prompts:
        sample = next(r for r in records if r["prompt_idx"] == p)
        preview = truncate(sample["prompt"].replace("\n", " "), 60)
        prompt_options_html += f'      <option value="{p}">Prompt {p}: {preview}</option>\n'

    # Build a JS mapping: traceIndex -> promptIdx for that trace
    # Traces are ordered: for each scale, for each prompt (matching build_interactive_plot)
    trace_prompt_map: list[int] = []
    for scale_val in unique_scales:
        for prompt_idx in unique_prompts:
            has_data = any(
                r["scale"] == scale_val and r["prompt_idx"] == prompt_idx
                for r in records
            )
            if has_data:
                trace_prompt_map.append(prompt_idx)
    trace_prompt_map_js = json.dumps(trace_prompt_map)

    # Inject HTML/CSS/JS for filtering controls and click-to-show detail panel
    detail_panel = f"""
<style>
  body {{ margin: 0; display: flex; flex-direction: column; }}
  #controls {{
    padding: 8px 16px;
    background: #f0f0f0;
    border-bottom: 1px solid #ddd;
    font-family: sans-serif;
    font-size: 13px;
    display: flex;
    align-items: center;
    gap: 16px;
    flex-shrink: 0;
  }}
  #controls label {{ font-weight: bold; color: #555; }}
  #controls select {{ padding: 4px 8px; font-size: 13px; max-width: 400px; }}
  #main-area {{ display: flex; flex: 1; }}
  .plotly-graph-div {{ flex-shrink: 0; }}
  #detail-panel {{
    width: 400px;
    height: 700px;
    overflow-y: auto;
    border-left: 2px solid #ddd;
    padding: 16px;
    font-family: sans-serif;
    font-size: 13px;
    box-sizing: border-box;
    background: #fafafa;
  }}
  #detail-panel h3 {{ margin-top: 0; color: #333; }}
  #detail-panel .field-label {{ font-weight: bold; color: #555; margin-top: 10px; }}
  #detail-panel .field-value {{ margin: 4px 0 0 0; white-space: pre-wrap; word-wrap: break-word; }}
  #detail-panel .placeholder {{ color: #999; font-style: italic; }}
</style>
<div id="controls">
  <label for="prompt-filter">Filter by prompt:</label>
  <select id="prompt-filter">
    {prompt_options_html}
  </select>
</div>
<div id="main-area">
  <div id="detail-panel">
    <h3>Point Details</h3>
    <p class="placeholder">Click a point to see full prompt and response text.</p>
  </div>
</div>
<script>
document.addEventListener('DOMContentLoaded', function() {{
  var plotEl = document.querySelector('.plotly-graph-div');
  var panel = document.getElementById('detail-panel');
  var mainArea = document.getElementById('main-area');
  if (!plotEl || !panel || !mainArea) return;

  // Move the plot div inside the flex container
  mainArea.insertBefore(plotEl, panel);

  // tracePromptMap[traceIndex] = promptIdx for that trace
  var tracePromptMap = {trace_prompt_map_js};
  var nTraces = tracePromptMap.length;

  // Track which traces the user has hidden via legend clicks
  // so prompt filter respects manual legend toggling.
  var legendHidden = new Array(nTraces).fill(false);

  // --- Prompt filtering via trace visibility ---
  var promptFilter = document.getElementById('prompt-filter');

  promptFilter.addEventListener('change', function() {{
    var selected = this.value;
    var visibility = [];
    for (var t = 0; t < nTraces; t++) {{
      if (legendHidden[t]) {{
        visibility.push(false);
      }} else if (selected === 'all') {{
        visibility.push(true);
      }} else {{
        visibility.push(String(tracePromptMap[t]) === selected);
      }}
    }}
    Plotly.restyle(plotEl, {{'visible': visibility}});
  }});

  // --- Click to show detail panel ---
  plotEl.on('plotly_click', function(data) {{
    if (!data.points.length) return;
    var pt = data.points[0];
    var cd = pt.customdata;
    var promptIdx = cd[0];
    var scale = cd[1];
    var respIdx = cd[2];
    var prompt = cd[3];
    var response = cd[4];

    panel.innerHTML =
      '<h3>Point Details</h3>' +
      '<div class="field-label">Prompt Index</div>' +
      '<div class="field-value">' + promptIdx + '</div>' +
      '<div class="field-label">Scale</div>' +
      '<div class="field-value">' + scale + '</div>' +
      '<div class="field-label">Response Index</div>' +
      '<div class="field-value">' + respIdx + '</div>' +
      '<div class="field-label">Prompt</div>' +
      '<div class="field-value">' + escapeHtml(prompt) + '</div>' +
      '<div class="field-label">Response</div>' +
      '<div class="field-value">' + escapeHtml(response) + '</div>';
  }});

  function escapeHtml(text) {{
    var div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }}
}});
</script>
"""
    with open(html_path, "r") as f:
        html_content = f.read()
    html_content = html_content.replace("</body>", detail_panel + "</body>")
    with open(html_path, "w") as f:
        f.write(html_content)

    print(f"Saved interactive UMAP plot to {html_path}")

    # 5. Write description file
    desc_path = output_dir / "interactive_umap.description.txt"
    desc_path.write_text(
        "Interactive UMAP Visualization of Steered LLM Responses\n"
        "========================================================\n\n"
        "This HTML file contains an interactive 2D UMAP projection of sentence\n"
        "embeddings (all-MiniLM-L6-v2) for LLM responses generated at different\n"
        "steering scales.\n\n"
        "- Each point represents one generated response.\n"
        "- Points are colored by the steering scale used during generation.\n"
        "- Hover over a point to see:\n"
        "    - prompt_idx: which prompt was used\n"
        "    - scale: the steering vector multiplier\n"
        "    - response_idx: which sample (among multiple per prompt+scale)\n"
        "    - A preview of the prompt text (~200 chars)\n"
        "    - A preview of the response text (~300 chars)\n\n"
        "Data source: outputs/happy_recon/responses.jsonl\n"
        "Embeddings: outputs/happy_recon/embeddings.npz\n"
        "UMAP parameters: n_neighbors=15, min_dist=0.1, seed=42\n"
    )
    print(f"Saved description to {desc_path}")


if __name__ == "__main__":
    main()
