"""Parsers for the route metadata used by the VIRL benchmark."""


def parse_navigation_string(value: str) -> list[dict[str, str]]:
    """Parse the line-oriented milestone dictionaries in SFTvsRL route files."""
    result = []
    for raw_block in value.split("}"):
        block = raw_block.lstrip("{").strip()
        if not block:
            continue
        milestone = {}
        for raw_line in block.splitlines():
            line = raw_line.strip().rstrip(",")
            if not line or ":" not in line:
                continue
            key, item = (part.strip() for part in line.split(":", 1))
            milestone[key.strip("\"'")] = item.strip("\"'")
        if milestone:
            result.append(milestone)
    return result
