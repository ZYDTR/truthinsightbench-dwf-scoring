#!/usr/bin/env python3
"""Validate Markdown, adaptive term cards, and a standalone HTML reader."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any


FIXED_CARD_FIELDS = re.compile(r"一句话|在这份\s*Skill\s*里|容易混淆", re.IGNORECASE)
MECHANICAL_CONTRAST = re.compile(r"(?:不是|并非)[\s\S]{0,100}而是")
EXTERNAL_SCRIPT = re.compile(r"<script[^>]+\bsrc\s*=", re.IGNORECASE)
EXTERNAL_STYLESHEET = re.compile(
    r"<link[^>]+\brel\s*=\s*['\"]?stylesheet", re.IGNORECASE
)
PAYLOAD = re.compile(
    r'<script\s+id="term-card-data"\s+type="application/json">(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)
ASCII_WORD = re.compile(r"[A-Za-z0-9_]")
SOURCE_FORM_LINE = re.compile(r"^(?:原文写法|来源写法)(?:（[^）]*）)?\s*[：:]", re.IGNORECASE)


def norm(value: str) -> str:
    return value.strip().casefold()


def resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def contains_alias(text: str, alias: str) -> bool:
    folded_text = text.casefold()
    folded_alias = alias.casefold()
    start = 0
    while True:
        index = folded_text.find(folded_alias, start)
        if index < 0:
            return False
        before = folded_text[index - 1] if index > 0 else ""
        after_index = index + len(folded_alias)
        after = folded_text[after_index] if after_index < len(folded_text) else ""
        first = folded_alias[0] if folded_alias else ""
        last = folded_alias[-1] if folded_alias else ""
        left_ok = not (ASCII_WORD.match(first) and ASCII_WORD.match(before))
        right_ok = not (ASCII_WORD.match(last) and ASCII_WORD.match(after))
        if left_ok and right_ok:
            return True
        start = index + 1


def point_of_use_text(markdown: str) -> str:
    """Return prose that is allowed to promise an interactive term card."""
    kept: list[str] = []
    in_fence = False
    for line in markdown.splitlines():
        if re.match(r"^\s*```", line):
            in_fence = not in_fence
            continue
        if in_fence or re.match(r"^\s*#{1,6}\s+", line) or SOURCE_FORM_LINE.match(line.strip()):
            continue
        kept.append(line)
    return "\n".join(kept)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_string(value: Any, field: str, errors: list[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field}: expected non-empty string")
        return ""
    return value.strip()


def owner_for_alias(cards: dict[str, Any], wanted: str) -> str | None:
    needle = norm(wanted)
    for key, card in cards.items():
        aliases = card.get("aliases", []) if isinstance(card, dict) else []
        if any(isinstance(alias, str) and norm(alias) == needle for alias in aliases):
            return key
    return None


def check_topology(name: str, key: str, topology: Any, errors: list[str]) -> None:
    field = f"{name}/{key}.topology"
    if not isinstance(topology, dict):
        errors.append(f"{field}: expected object for composite card")
        return
    require_string(topology.get("caption"), f"{field}.caption", errors)
    nodes = topology.get("nodes")
    if not isinstance(nodes, list) or not 2 <= len(nodes) <= 8:
        errors.append(f"{field}.nodes: expected 2-8 nodes")
        nodes = []
    node_ids: set[str] = set()
    positions: set[tuple[int, int]] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"{field}.nodes[{index}]: expected object")
            continue
        node_id = require_string(node.get("id"), f"{field}.nodes[{index}].id", errors)
        require_string(node.get("label"), f"{field}.nodes[{index}].label", errors)
        if node_id and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", node_id):
            errors.append(f"{field}.nodes[{index}].id: invalid identifier")
        if node_id in node_ids:
            errors.append(f"{field}: duplicate node id {node_id!r}")
        if node_id:
            node_ids.add(node_id)
        row = node.get("row")
        column = node.get("column")
        if not isinstance(row, int) or row < 0 or not isinstance(column, int) or column < 0:
            errors.append(f"{field}.nodes[{index}]: row/column must be non-negative integers")
        else:
            position = (row, column)
            if position in positions:
                errors.append(f"{field}: duplicate grid position {position}")
            positions.add(position)

    edges = topology.get("edges")
    if not isinstance(edges, list) or not edges:
        errors.append(f"{field}.edges: expected at least one edge")
        edges = []
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            errors.append(f"{field}.edges[{index}]: expected object")
            continue
        source = require_string(edge.get("from"), f"{field}.edges[{index}].from", errors)
        target = require_string(edge.get("to"), f"{field}.edges[{index}].to", errors)
        require_string(edge.get("label"), f"{field}.edges[{index}].label", errors)
        if source and source not in node_ids:
            errors.append(f"{field}.edges[{index}]: unknown source node {source!r}")
        if target and target not in node_ids:
            errors.append(f"{field}.edges[{index}]: unknown target node {target!r}")
        if source and source == target:
            errors.append(f"{field}.edges[{index}]: self-edge is not allowed")

    groups = topology.get("groups")
    if not isinstance(groups, list) or not groups:
        errors.append(f"{field}.groups: expected at least one current-term boundary")
        groups = []
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            errors.append(f"{field}.groups[{index}]: expected object")
            continue
        require_string(group.get("label"), f"{field}.groups[{index}].label", errors)
        members = group.get("members")
        if not isinstance(members, list) or len(members) < 2:
            errors.append(f"{field}.groups[{index}].members: expected at least two node ids")
            continue
        for member in members:
            if not isinstance(member, str) or member not in node_ids:
                errors.append(f"{field}.groups[{index}]: unknown member {member!r}")


def check_html(
    name: str,
    html_path: Path,
    source_cards: dict[str, Any],
    errors: list[str],
) -> int:
    if not html_path.is_file():
        errors.append(f"{name}: HTML missing: {html_path}")
        return 0
    html = html_path.read_text(encoding="utf-8")
    if not re.search(r"<!doctype html>", html, re.IGNORECASE):
        errors.append(f"{name}: standalone HTML doctype missing")
    if EXTERNAL_SCRIPT.search(html):
        errors.append(f"{name}: external script found")
    if EXTERNAL_STYLESHEET.search(html):
        errors.append(f"{name}: external stylesheet found")
    if 'id="article-body"' not in html or 'id="term-card-data"' not in html:
        errors.append(f"{name}: reader article or term-card payload missing")

    match = PAYLOAD.search(html)
    if not match:
        errors.append(f"{name}: embedded term-card payload cannot be found")
        return len(html)
    try:
        embedded = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        errors.append(f"{name}: embedded term-card payload is invalid JSON: {exc}")
        return len(html)
    embedded_cards = embedded.get("cards") if isinstance(embedded, dict) else None
    if not isinstance(embedded_cards, dict):
        errors.append(f"{name}: embedded cards must be an object map")
        return len(html)
    if set(embedded_cards) != set(source_cards):
        errors.append(f"{name}: HTML and source JSON card keys differ")
    for key in set(embedded_cards) & set(source_cards):
        for field in ("label", "aliases", "mentions", "structure", "topology", "body"):
            default = [] if field == "mentions" else None
            if embedded_cards[key].get(field, default) != source_cards[key].get(field, default):
                errors.append(f"{name}/{key}: HTML {field} differs from source JSON")
    composite_count = sum(
        1 for card in source_cards.values()
        if isinstance(card, dict) and card.get("structure") == "composite"
    )
    if composite_count and html.count('class=\\"term-topology\\"') < composite_count:
        errors.append(f"{name}: rendered topology count is smaller than composite card count")
    return len(html)


def check_document(spec: dict[str, Any], base: Path) -> dict[str, Any]:
    errors: list[str] = []
    name = require_string(spec.get("name"), "document.name", errors) or "unnamed"
    markdown_path = resolve(base, require_string(spec.get("markdown"), f"{name}.markdown", errors))
    cards_path = resolve(base, require_string(spec.get("cards"), f"{name}.cards", errors))
    html_value = spec.get("html")
    html_path = resolve(base, html_value) if isinstance(html_value, str) else None

    if not markdown_path.is_file():
        errors.append(f"{name}: Markdown missing: {markdown_path}")
        markdown = ""
    else:
        markdown = markdown_path.read_text(encoding="utf-8")
    if not cards_path.is_file():
        errors.append(f"{name}: cards JSON missing: {cards_path}")
        raw: Any = {}
    else:
        try:
            raw = load_json(cards_path)
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"{name}: cards JSON cannot be read: {exc}")
            raw = {}

    cards = raw.get("cards") if isinstance(raw, dict) else None
    if not isinstance(cards, dict):
        errors.append(f"{name}: cards must be an object map")
        cards = {}
    expected = spec.get("expected_cards")
    if isinstance(expected, int) and len(cards) != expected:
        errors.append(f"{name}: expected {expected} cards, found {len(cards)}")

    alias_owners: dict[str, str] = {}
    label_owners: dict[str, str] = {}
    body_owners: dict[str, str] = {}
    body_lengths: list[int] = []
    composite_cards = 0
    interactive_markdown = point_of_use_text(markdown)
    for key, card in cards.items():
        if not isinstance(card, dict):
            errors.append(f"{name}/{key}: card must be an object")
            continue
        label = require_string(card.get("label"), f"{name}/{key}.label", errors)
        body = require_string(card.get("body"), f"{name}/{key}.body", errors)
        structure = require_string(card.get("structure"), f"{name}/{key}.structure", errors)
        if structure not in {"atomic", "composite"}:
            errors.append(f"{name}/{key}.structure: expected atomic or composite")
        if structure == "composite":
            composite_cards += 1
            check_topology(name, key, card.get("topology"), errors)
        elif card.get("topology") is not None:
            errors.append(f"{name}/{key}: atomic card must not contain topology")
        aliases = card.get("aliases")
        if not isinstance(aliases, list) or len(aliases) != 1:
            errors.append(f"{name}/{key}.aliases: expected exactly one point-of-use term")
            aliases = []
        mentions = card.get("mentions", [])
        if not isinstance(mentions, list):
            errors.append(f"{name}/{key}.mentions: expected array when present")
            mentions = []
        for index, mention in enumerate(mentions):
            require_string(mention, f"{name}/{key}.mentions[{index}]", errors)
        clean_aliases: list[str] = []
        for index, alias in enumerate(aliases):
            clean = require_string(alias, f"{name}/{key}.aliases[{index}]", errors)
            if not clean:
                continue
            clean_aliases.append(clean)
            token = norm(clean)
            previous = alias_owners.get(token)
            if previous and previous != key:
                errors.append(f"{name}: alias {clean!r} belongs to both {previous} and {key}")
            alias_owners[token] = key
        if clean_aliases and not any(contains_alias(interactive_markdown, alias) for alias in clean_aliases):
            errors.append(f"{name}/{key}: no point-of-use alias appears in Markdown")
        label_token = norm(label) if label else ""
        if label_token:
            previous_label = label_owners.get(label_token)
            if previous_label and previous_label != key:
                errors.append(f"{name}: label {label!r} is reused by {previous_label} and {key}")
            label_owners[label_token] = key
        body_token = re.sub(r"\s+", " ", body).strip().casefold() if body else ""
        if body_token:
            previous_body = body_owners.get(body_token)
            if previous_body and previous_body != key:
                errors.append(f"{name}: body is duplicated by {previous_body} and {key}")
            body_owners[body_token] = key
        if FIXED_CARD_FIELDS.search(body):
            errors.append(f"{name}/{key}: fixed legacy card field found")
        if MECHANICAL_CONTRAST.search(body):
            errors.append(f"{name}/{key}: mechanical contrast construction found")
        if body:
            body_lengths.append(len(body))

    for forbidden in spec.get("forbidden_terms", []):
        needle = norm(str(forbidden))
        for key, card in cards.items():
            label = norm(str(card.get("label", ""))) if isinstance(card, dict) else ""
            aliases = card.get("aliases", []) if isinstance(card, dict) else []
            if needle == norm(key) or needle == label or any(
                isinstance(alias, str) and needle == norm(alias) for alias in aliases
            ):
                errors.append(f"{name}: forbidden card or alias remains: {forbidden}")
                break

    for required in spec.get("required_terms", []):
        if owner_for_alias(cards, str(required)) is None:
            errors.append(f"{name}: required term card missing: {required}")

    for required in spec.get("topology_terms", []):
        owner = owner_for_alias(cards, str(required))
        if owner is None:
            errors.append(f"{name}: topology term card missing: {required}")
        elif cards[owner].get("structure") != "composite" or not isinstance(cards[owner].get("topology"), dict):
            errors.append(f"{name}: topology term is not a composite card with topology: {required}")

    if spec.get("merged_alias_groups"):
        errors.append(f"{name}: merged_alias_groups is obsolete; use distinct_term_groups")

    for group in spec.get("distinct_term_groups", []):
        if not isinstance(group, list) or len(group) < 2:
            errors.append(f"{name}: distinct_term_groups entries need at least two terms")
            continue
        owners = [owner_for_alias(cards, str(alias)) for alias in group]
        if any(owner is None for owner in owners):
            errors.append(f"{name}: distinct group has missing terms: {group}")
        elif len(set(owners)) != len(owners):
            errors.append(f"{name}: distinct group incorrectly shares a card: {group}")

    html_chars = check_html(name, html_path, cards, errors) if html_path else None
    sorted_lengths = sorted(body_lengths)
    stats = {
        "cards": len(cards),
        "aliases": len(alias_owners),
        "composite_cards": composite_cards,
        "min_body_chars": sorted_lengths[0] if sorted_lengths else 0,
        "median_body_chars": statistics.median(sorted_lengths) if sorted_lengths else 0,
        "max_body_chars": sorted_lengths[-1] if sorted_lengths else 0,
        "html_chars": html_chars,
    }
    return {"name": name, "status": "PASS" if not errors else "FAIL", "stats": stats, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="bundle validation manifest JSON")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_json(manifest_path)
    documents = manifest.get("documents") if isinstance(manifest, dict) else None
    if not isinstance(documents, list) or not documents:
        raise SystemExit("manifest.documents must be a non-empty array")
    results = [check_document(spec, manifest_path.parent) for spec in documents]
    status = "PASS" if all(item["status"] == "PASS" for item in results) else "FAIL"
    report = {"status": status, "documents": results}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"EXPLAINER_BUNDLE: {status}")
        for item in results:
            print(f"{item['name']}: {item['status']} {json.dumps(item['stats'], ensure_ascii=False)}")
            for error in item["errors"]:
                print(f"  - {error}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
