import re

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous",
    r"reveal\s+(all\s+)?(secrets?|keys?|credentials?|passwords?|config)",
    r"override\s+(instructions?|rules?|permissions?|system)",
    r"forget\s+(your\s+)?(instructions?|rules?|context)",
    r"print\s+(your\s+)?(system\s+prompt|instructions?)",
    r"act\s+as\s+(if|a|an|though)",
    r"jailbreak",
    r"you\s+are\s+now\s+(a|an|the|DAN)",
    r"disregard\s+(your\s+)?(previous|prior|all)",
    r"repeat\s+(after\s+me|the\s+following|everything)",
    r"show\s+me\s+(the\s+)?(system\s+prompt|hidden\s+instructions?)",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def sanitise_input(text: str) -> str:
    text = text.strip()
    text = text[:1000]
    # Escape angle brackets that could break XML-tag wrapping
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    return text


def detect_injection(text: str) -> bool:
    for pattern in _COMPILED:
        if pattern.search(text):
            return True
    return False
