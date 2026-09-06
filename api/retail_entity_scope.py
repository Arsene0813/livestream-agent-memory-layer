"""Store identifiers in requests and facts for the current retail datasets."""

import re


def extract_retail_entity_ids(message: str) -> list[str]:
    q = (message or "").lower()
    suffixes: list[str] = []
    token = r"(?:[a-z]\d*|\d+)"

    def add(value: str) -> None:
        if value not in suffixes:
            suffixes.append(value)

    for match in re.finditer(r"(?<![a-z0-9_])store_([a-z0-9][a-z0-9_]*)", q):
        add(match.group(1))
    for match in re.finditer(rf"(?<![a-z0-9_])stores?\s+({token})(?![a-z0-9_])", q):
        add(match.group(1))
        tail = q[match.end():]
        interval = re.match(r"\s*(?:-|–|—|to\s+|through\s+)([a-z])(?![a-z0-9_])", tail)
        if interval and re.fullmatch(r"[a-z]", match.group(1)):
            first, last = sorted((ord(match.group(1)), ord(interval.group(1))))
            for number in range(first, last + 1):
                add(chr(number))
            tail = tail[interval.end():]
        while True:
            item = re.match(
                rf"\s*(?:,\s*(?:and\s+)?|and\s+|&\s*)(?:store\s+)?({token})(?![a-z0-9_])",
                tail,
            )
            if not item:
                break
            add(item.group(1))
            tail = tail[item.end():]
    for match in re.finditer(rf"(?<![a-z0-9_])({token})店", q):
        add(match.group(1))
    for match in re.finditer(rf"(?:门店|店铺)\s*({token})(?![a-z0-9_])", q):
        add(match.group(1))
    for match in re.finditer(r"(?<![a-z0-9_])Store([A-Z]\d*|\d+)(?![a-zA-Z0-9_])", message or ""):
        add(match.group(1).lower())
    if re.search(r"(?<![a-z0-9_-])b\s*-\s*f(?![a-z0-9_-])", q):
        for suffix in "bcdef":
            add(suffix)
    return ["store_" + suffix for suffix in suffixes]


def resolve_retail_entity_id(message: str, entity_id: str | None) -> str:
    raw = (entity_id or "").strip().lower().replace(" ", "_")
    if raw:
        return "store_" + raw if re.fullmatch(r"[a-z]|\d+", raw) else raw
    mentioned = extract_retail_entity_ids(message)
    return mentioned[0] if len(mentioned) == 1 else ""


def retail_fact_matches_entities(fact: object, entity_ids: set[str]) -> bool:
    if not isinstance(fact, dict):
        return False
    canonical_ids = {"store_" + value.removeprefix("store_").upper() for value in entity_ids}
    entity = fact.get("entity_id")
    if not isinstance(entity, str) or entity not in canonical_ids:
        return False
    return (
        fact.get("entity_id_norm", entity.lower()) == entity.lower()
        and ("store_id" not in fact or fact["store_id"] == entity.removeprefix("store_"))
        and fact.get("domain", "retail_ops") == "retail_ops"
        and fact.get("kind") == "retail_memory_fact"
        and fact.get("type") == "retail_metric_profile"
        and fact.get("is_active") is True
    )
