# docs/

Living documentation for the Steering Diversity project.

## Files

| File | Purpose |
|------|---------|
| `paper.md` | **Living research paper** — the main document. Edit this directly. |
| `steering_diversity_write_up.md` | Original draft writeup (kept for reference; content migrated to `paper.md`) |
| `steering_diversity_write_up.pdf` | PDF export of original draft |
| `PREREGISTRATION.md` | Pre-registered statistical analysis plan (canonical record; included in `paper.md` Appendix A) |
| `RESEARCH_LOG.md` | Chronological experiment log (canonical record; results incorporated into `paper.md` Section 6) |

## Workflow

### Editing the paper

- `paper.md` is the living document — edit it directly
- Use tag conventions to mark incomplete sections:
  - `<!-- TODO: description -->` — work needed to complete this section
  - `<!-- NOTE: description -->` — informal notes, Claude chat links, planning items
  - `<!-- FEEDBACK: question -->` — questions for mentors/collaborators
- Find all gaps: `grep -c 'TODO' docs/paper.md` or search for `<!-- TODO`

### Figures

Figures are referenced with relative paths from `docs/`:

```markdown
![Caption](../outputs/happy_recon/plots/within_vs_pooled_diversity.png)
```

Figures live in `outputs/*/plots/` (steering experiments), `eval_awareness/figures/` (eval awareness), and `icl-diversity/figures/` (ICL diversity metric).

### Conversion to LaTeX

When ready for submission, convert via pandoc or manual rewrite:

```bash
pandoc docs/paper.md -o paper.tex --template=template.tex
```

### Relationship to submodule docs

Submodules (`eval_awareness/`, `icl-diversity/`, `Test_Awareness_Steering/`, `rlfh-gen-div/`) have their own READMEs and docs. `paper.md` summarizes their methodology and references them; it does not duplicate their full documentation.
