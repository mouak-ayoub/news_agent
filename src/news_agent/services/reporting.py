from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
import re

from ..models.triage import SourceProfile
from ..models.triage import TriageBrief


def render_html_report(brief: TriageBrief) -> str:
    profile_map = {profile.name: profile for profile in brief.source_profiles}
    findings = _render_findings(brief, profile_map)
    numbers = _render_numbers(brief)
    claims = _render_claims(brief)
    facts = _render_fact_blocks(brief)
    framing = _render_list(brief.framing_analysis, "No framing analysis available yet.")
    history = _render_list(brief.historical_context, "No historical context available yet.")
    uncertainties = _render_list(brief.uncertainties, "No explicit uncertainties recorded.")
    countries = len({country for country in brief.entities.countries if country})
    numeric_entries = sum(len(finding.reported_numbers) for finding in brief.source_findings)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{escape(brief.query)} | News Agent Report</title>
  <style>
    :root {{
      --bg: #f4efe8;
      --panel: rgba(255, 251, 245, 0.9);
      --panel-strong: rgba(255, 255, 255, 0.98);
      --ink: #181513;
      --muted: #5f574f;
      --line: rgba(24, 21, 19, 0.1);
      --accent: #b24c1c;
      --accent-soft: rgba(178, 76, 28, 0.14);
      --teal: #1c6670;
      --teal-soft: rgba(28, 102, 112, 0.12);
      --gold: #ba8a15;
      --gold-soft: rgba(186, 138, 21, 0.14);
      --shadow: 0 24px 80px rgba(23, 18, 14, 0.12);
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(178, 76, 28, 0.18), transparent 34%),
        radial-gradient(circle at top right, rgba(28, 102, 112, 0.14), transparent 30%),
        linear-gradient(180deg, #f8f4ee 0%, #eee5d8 100%);
      min-height: 100vh;
    }}

    .page {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 38px 24px 60px;
    }}

    .hero {{
      position: relative;
      overflow: hidden;
      padding: 34px;
      border-radius: 30px;
      border: 1px solid var(--line);
      background: linear-gradient(140deg, rgba(255,255,255,0.94), rgba(250,242,230,0.92));
      box-shadow: var(--shadow);
    }}

    .hero::after {{
      content: "";
      position: absolute;
      right: -80px;
      bottom: -80px;
      width: 260px;
      height: 260px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(178, 76, 28, 0.22), transparent 64%);
    }}

    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }}

    h1 {{
      margin: 16px 0 10px;
      max-width: 18ch;
      font-size: clamp(2.2rem, 4.8vw, 4.6rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}

    .hero p {{
      max-width: 760px;
      margin: 0;
      font-size: 1.02rem;
      line-height: 1.75;
      color: var(--muted);
    }}

    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-top: 22px;
    }}

    .stat {{
      padding: 18px;
      border-radius: 22px;
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid var(--line);
    }}

    .stat .label {{
      display: block;
      margin-bottom: 8px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
    }}

    .stat .value {{
      font-size: 2rem;
      font-weight: 750;
      letter-spacing: -0.04em;
    }}

    .layout {{
      display: grid;
      grid-template-columns: 1.4fr 0.9fr;
      gap: 18px;
      margin-top: 22px;
    }}

    .stack {{
      display: grid;
      gap: 18px;
    }}

    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(8px);
    }}

    .card h2 {{
      margin: 0 0 16px;
      font-size: 0.95rem;
      font-weight: 800;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
    }}

    .finding-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
    }}

    .finding {{
      display: grid;
      gap: 12px;
      padding: 18px;
      border-radius: 22px;
      background: var(--panel-strong);
      border: 1px solid var(--line);
    }}

    .finding-title {{
      margin: 0;
      font-size: 1.1rem;
      line-height: 1.25;
    }}

    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}

    .pill {{
      display: inline-flex;
      align-items: center;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      border: 1px solid transparent;
    }}

    .pill.teal {{
      background: var(--teal-soft);
      color: var(--teal);
      border-color: rgba(28, 102, 112, 0.12);
    }}

    .pill.gold {{
      background: var(--gold-soft);
      color: #7b5a08;
      border-color: rgba(186, 138, 21, 0.12);
    }}

    .pill.accent {{
      background: var(--accent-soft);
      color: var(--accent);
      border-color: rgba(178, 76, 28, 0.12);
    }}

    .finding blockquote {{
      margin: 0;
      padding: 14px 16px;
      border-left: 3px solid var(--accent);
      border-radius: 16px;
      background: rgba(178, 76, 28, 0.06);
      color: var(--ink);
    }}

    .finding p {{
      margin: 0;
      line-height: 1.65;
      color: var(--muted);
    }}

    .finding a {{
      color: var(--teal);
      text-decoration: none;
      font-weight: 700;
    }}

    .finding a:hover {{
      text-decoration: underline;
    }}

    .numbers {{
      display: grid;
      gap: 12px;
    }}

    .number-card {{
      padding: 16px;
      border-radius: 20px;
      background: var(--panel-strong);
      border: 1px solid var(--line);
    }}

    .number-card strong {{
      display: block;
      margin-bottom: 10px;
      font-size: 1rem;
    }}

    .number-line {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}

    .claims {{
      display: grid;
      gap: 12px;
    }}

    .claim {{
      padding: 16px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid var(--line);
    }}

    .claim strong {{
      display: block;
      margin-bottom: 8px;
      line-height: 1.5;
    }}

    .reasoning {{
      display: grid;
      gap: 12px;
    }}

    .reasoning-box {{
      padding: 18px;
      border-radius: 20px;
      border: 1px solid var(--line);
    }}

    .reasoning-box.observation {{
      background: rgba(28, 102, 112, 0.08);
    }}

    .reasoning-box.inference {{
      background: rgba(186, 138, 21, 0.12);
    }}

    .reasoning-box.speculation {{
      background: rgba(178, 76, 28, 0.1);
    }}

    ul {{
      margin: 0;
      padding-left: 18px;
      line-height: 1.7;
    }}

    .footer {{
      margin-top: 18px;
      text-align: right;
      color: var(--muted);
      font-size: 0.9rem;
    }}

    @media (max-width: 980px) {{
      .stats {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}

      .layout {{
        grid-template-columns: 1fr;
      }}
    }}

    @media (max-width: 640px) {{
      .page {{
        padding: 24px 14px 38px;
      }}

      .hero, .card {{
        border-radius: 22px;
        padding: 18px;
      }}

      .stats {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <span class="eyebrow">Dynamic News Triage</span>
      <h1>{escape(brief.query)}</h1>
      <p>{escape(brief.final_brief or "No final brief available yet.")}</p>
      <div class="stats">
        <div class="stat">
          <span class="label">Sources Read</span>
          <span class="value">{len(brief.source_findings)}</span>
        </div>
        <div class="stat">
          <span class="label">Countries Seen</span>
          <span class="value">{countries}</span>
        </div>
        <div class="stat">
          <span class="label">Numeric Claims</span>
          <span class="value">{numeric_entries}</span>
        </div>
        <div class="stat">
          <span class="label">Main Claims</span>
          <span class="value">{len(brief.main_claims)}</span>
        </div>
      </div>
    </section>

    <section class="layout">
      <div class="stack">
        <article class="card">
          <h2>Outlet Findings</h2>
          <div class="finding-grid">{findings}</div>
        </article>

        <article class="card">
          <h2>Main Claims</h2>
          <div class="claims">{claims}</div>
        </article>

        <article class="card">
          <h2>Reasoning Layers</h2>
          <div class="reasoning">{facts}</div>
        </article>
      </div>

      <div class="stack">
        <article class="card">
          <h2>Reported Numbers</h2>
          <div class="numbers">{numbers}</div>
        </article>

        <article class="card">
          <h2>Framing Analysis</h2>
          {framing}
        </article>

        <article class="card">
          <h2>Historical Context</h2>
          {history}
        </article>

        <article class="card">
          <h2>Uncertainties</h2>
          {uncertainties}
        </article>
      </div>
    </section>

    <div class="footer">Generated {escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}</div>
  </div>
</body>
</html>
"""


def default_report_path(query: str, base_dir: str | Path) -> Path:
    root = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / "reports" / f"{timestamp}_{_slugify(query)}.html"


def write_html_report(brief: TriageBrief, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html_report(brief), encoding="utf-8")
    return path


def _render_findings(brief: TriageBrief, profile_map: dict[str, SourceProfile]) -> str:
    if not brief.source_findings:
        return _empty_card("No outlet findings were extracted for this query.")

    cards = []
    for finding in brief.source_findings:
        profile = profile_map.get(finding.outlet_name)
        numbers = (
            "".join(
                f'<span class="pill gold">{escape(number)}</span>'
                for number in finding.reported_numbers
            )
            if finding.reported_numbers
            else '<span class="pill gold">No explicit figure</span>'
        )
        profile_bits = []
        if finding.country:
            profile_bits.append(f'<span class="pill teal">{escape(finding.country)}</span>')
        if profile:
            if profile.type:
                profile_bits.append(f'<span class="pill teal">{escape(profile.type)}</span>')
            if profile.orientation:
                profile_bits.append(f'<span class="pill accent">{escape(profile.orientation)}</span>')
        source_position = escape(finding.source_position or "No clear position extracted.")
        judgment = escape(finding.judgment or "No judgment available.")
        notes = escape(finding.notes or "No notes captured.")
        url = escape(finding.url)
        link = (
            f'<a href="{url}" target="_blank" rel="noreferrer">Open source</a>'
            if finding.url
            else '<span class="pill accent">No source link</span>'
        )
        cards.append(
            "<article class=\"finding\">"
            f"<div><div class=\"meta\">{''.join(profile_bits)}</div>"
            f"<h3 class=\"finding-title\">{escape(finding.outlet_name or 'Unknown outlet')}</h3>"
            f"<p>{escape(finding.headline or 'No headline captured.')}</p></div>"
            f"<blockquote>{source_position}</blockquote>"
            f"<div><div class=\"meta\">{numbers}</div></div>"
            f"<p><strong>Judgment:</strong> {judgment}</p>"
            f"<p><strong>Notes:</strong> {notes}</p>"
            f"{link}"
            "</article>"
        )
    return "".join(cards)


def _render_numbers(brief: TriageBrief) -> str:
    with_numbers = [finding for finding in brief.source_findings if finding.reported_numbers]
    if not with_numbers:
        return _empty_card("No explicit figures were extracted from the retrieved sources.")

    cards = []
    for finding in with_numbers:
        lines = "".join(
            f'<span class="pill gold">{escape(number)}</span>'
            for number in finding.reported_numbers
        )
        cards.append(
            f'<div class="number-card"><strong>{escape(finding.outlet_name)}</strong>'
            f'<div class="number-line">{lines}</div>'
            f'<p>{escape(finding.judgment or "Figure present but still requires cross-checking.")}</p></div>'
        )
    return "".join(cards)


def _render_claims(brief: TriageBrief) -> str:
    if not brief.main_claims:
        return _empty_card("No main claim extracted yet.")

    blocks = []
    for claim in brief.main_claims:
        blocks.append(
            f'<div class="claim"><strong>{escape(claim.claim)}</strong>'
            f'<div class="meta">'
            f'<span class="pill accent">{escape(claim.status)}</span>'
            f'<span class="pill gold">evidence: {escape(claim.evidence_level)}</span>'
            f"</div></div>"
        )
    return "".join(blocks)


def _render_fact_blocks(brief: TriageBrief) -> str:
    return (
        f'<section class="reasoning-box observation"><h3>Observation</h3>{_render_list(brief.fact_inference_speculation.observation, "No direct observations recorded.")}</section>'
        f'<section class="reasoning-box inference"><h3>Evidence-Backed Inference</h3>{_render_list(brief.fact_inference_speculation.evidence_backed_inference, "No inferences recorded.")}</section>'
        f'<section class="reasoning-box speculation"><h3>Speculation</h3>{_render_list(brief.fact_inference_speculation.speculation, "No speculation recorded.")}</section>'
    )


def _render_list(items: list[str], empty_text: str) -> str:
    if not items:
        return f"<p>{escape(empty_text)}</p>"
    return "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ul>"


def _empty_card(text: str) -> str:
    return f'<div class="number-card"><p>{escape(text)}</p></div>'


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return normalized[:80] or "report"
