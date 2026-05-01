from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
import re
from string import Template

from ..models.triage import SourceProfile
from ..models.triage import TriageBrief


NUMBER_PATTERN = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?(?:\s?(?:k|m|b|million|billion|thousand|percent|%))?\b",
    re.IGNORECASE,
)


def render_html_report(brief: TriageBrief) -> str:
    """Render a report using the external HTML template in config/html."""
    profile_map = {profile.name: profile for profile in brief.source_profiles}
    countries = len({country for country in brief.entities.countries if country})
    numeric_entries = sum(len(finding.reported_numbers) for finding in brief.source_findings)

    return _load_report_template().safe_substitute(
        query=escape(brief.query),
        final_brief=escape(brief.final_brief or "No final brief available yet."),
        source_count=str(len(brief.source_findings)),
        country_count=str(countries),
        numeric_entries=str(numeric_entries),
        claim_count=str(len(brief.main_claims)),
        findings=_render_findings(brief, profile_map),
        numbers=_render_numbers(brief),
        claims=_render_claims(brief),
        facts=_render_fact_blocks(brief),
        framing=_render_list(brief.framing_analysis, "No framing analysis available yet."),
        history=_render_list(brief.historical_context, "No historical context available yet."),
        uncertainties=_render_list(brief.uncertainties, "No explicit uncertainties recorded."),
        generated_at=escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )


def default_report_path(query: str, base_dir: str | Path) -> Path:
    root = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / "reports" / f"{timestamp}_{_slugify(query)}.html"


def write_html_report(brief: TriageBrief, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html_report(brief), encoding="utf-8")
    return path


def _load_report_template() -> Template:
    """Load the HTML shell so visual layout can change without Python edits."""
    return Template(_report_template_path().read_text(encoding="utf-8"))


def _report_template_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "html" / "report.html"


def _render_findings(brief: TriageBrief, profile_map: dict[str, SourceProfile]) -> str:
    """Render one card for each source finding returned by summarization."""
    if not brief.source_findings:
        return _empty_card("No outlet findings were extracted for this query.")

    cards = []
    for finding in brief.source_findings:
        profile = profile_map.get(finding.outlet_name)
        numbers = _render_number_pills(finding.reported_numbers)
        profile_bits = _render_profile_pills(finding.country, profile)
        url = escape(finding.url)
        link = (
            f'<a href="{url}" target="_blank" rel="noreferrer">Open source</a>'
            if finding.url
            else '<span class="pill accent">No source link</span>'
        )
        cards.append(
            '<article class="finding">'
            f'<div><div class="meta">{profile_bits}</div>'
            f'<h3 class="finding-title">{escape(finding.outlet_name or "Unknown outlet")}</h3>'
            f'<p>{escape(finding.headline or "No headline captured.")}</p></div>'
            f'<blockquote>{escape(finding.source_position or "No clear position extracted.")}</blockquote>'
            f'<div class="meta">{numbers}</div>'
            f'<p><strong>Judgment:</strong> {escape(finding.judgment or "No judgment available.")}</p>'
            f'<p><strong>Notes:</strong> {escape(finding.notes or "No notes captured.")}</p>'
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
        cards.append(
            f'<div class="number-card"><strong>{escape(finding.outlet_name)}</strong>'
            f'<div class="number-line">{_render_number_pills(finding.reported_numbers)}</div>'
            f'<p>{escape(finding.judgment or "Figure present but still requires cross-checking.")}</p></div>'
        )
    return "".join(cards)


def _render_claims(brief: TriageBrief) -> str:
    if not brief.main_claims:
        return _empty_card("No main claim extracted yet.")

    return "".join(
        f'<div class="claim"><strong>{escape(claim.claim)}</strong>'
        f'<div class="meta">'
        f'<span class="pill accent">{escape(claim.status)}</span>'
        f'<span class="pill gold">evidence: {escape(claim.evidence_level)}</span>'
        f"</div></div>"
        for claim in brief.main_claims
    )


def _render_fact_blocks(brief: TriageBrief) -> str:
    facts = brief.fact_inference_speculation
    return (
        '<section class="reasoning-box observation"><h3>Observation</h3>'
        f'{_render_list(facts.observation, "No direct observations recorded.")}</section>'
        '<section class="reasoning-box inference"><h3>Evidence-Backed Inference</h3>'
        f'{_render_list(facts.evidence_backed_inference, "No inferences recorded.")}</section>'
        '<section class="reasoning-box speculation"><h3>Speculation</h3>'
        f'{_render_list(facts.speculation, "No speculation recorded.")}</section>'
    )


def _render_profile_pills(country: str, profile: SourceProfile | None) -> str:
    pills = []
    if country:
        pills.append(f'<span class="pill teal">{escape(country)}</span>')
    if profile:
        if profile.type:
            pills.append(f'<span class="pill teal">{escape(profile.type)}</span>')
        if profile.orientation:
            pills.append(f'<span class="pill accent">{escape(profile.orientation)}</span>')
    return "".join(pills)


def _render_number_pills(numbers: list[str]) -> str:
    if not numbers:
        return '<span class="pill gold">No explicit figure</span>'
    return "".join(f'<span class="pill gold">{escape(number)}</span>' for number in numbers)


def _render_list(items: list[str], empty_text: str) -> str:
    if not items:
        return f"<p>{escape(empty_text)}</p>"
    return "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ul>"


def _empty_card(text: str) -> str:
    return f'<div class="number-card"><p>{escape(text)}</p></div>'


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return normalized[:80] or "report"
