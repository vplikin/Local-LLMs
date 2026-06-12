import json
import re

from .strip import strip, extract_json


def _json_exact(t, obj):
    try:
        return json.loads(extract_json(t).strip()) == obj
    except Exception:
        return False


def _word_count(t):
    return len([w for w in strip(t).split() if w.strip()])


CASES = [
    {"id": "json_status",
     "prompt": 'Return a JSON object with exactly one key "status" whose value is "ok". Output only the JSON.',
     "check": lambda t: _json_exact(t, {"status": "ok"})},
    {"id": "word_limit_10",
     "prompt": "Describe the ocean in 10 words or fewer.",
     "check": lambda t: 0 < _word_count(t) <= 10},
    {"id": "no_letter_e",
     "prompt": "Write one sentence about cars without using the letter 'e'.",
     "check": lambda t: t.strip() != "" and "e" not in t.lower()},
    {"id": "date_format",
     "prompt": "Output exactly the date 2025-01-01 in YYYY-MM-DD format and nothing else.",
     "check": lambda t: bool(re.fullmatch(r"\s*\d{4}-\d{2}-\d{2}\s*", t))},
    {"id": "uppercase_hello",
     "prompt": "Reply with the word HELLO in all capital letters and nothing else.",
     "check": lambda t: t.strip() == "HELLO" or strip(t).strip() == "HELLO"},
    {"id": "exactly_three_bullets",
     "prompt": "List exactly three primary colors as a bulleted list using '-' for each bullet.",
     "check": lambda t: len([l for l in t.splitlines() if l.strip().startswith("-")]) == 3},
    {"id": "two_paragraphs",
     "prompt": "Write exactly two paragraphs about rain, separated by one blank line.",
     "check": lambda t: len([p for p in re.split(r"\n\s*\n", t.strip()) if p.strip()]) == 2},
    {"id": "starts_with_word",
     "prompt": "Write a sentence about dogs that begins with the word 'Loyal'.",
     "check": lambda t: t.strip().startswith("Loyal") or strip(t).strip().startswith("Loyal")},
    {"id": "no_commas",
     "prompt": "Write a sentence about the sea that contains no commas.",
     "check": lambda t: t.strip() != "" and "," not in t},
    {"id": "ends_with_period",
     "prompt": "Write one short statement that ends with a period.",
     "check": lambda t: t.strip().endswith(".")},
    {"id": "keyword_present",
     "prompt": "Write a sentence about networking that includes the word 'latency'.",
     "check": lambda t: "latency" in strip(t).lower()},
    {"id": "number_range",
     "prompt": "Output a single integer between 50 and 60 inclusive, nothing else.",
     "check": lambda t: t.strip().isdigit() and 50 <= int(t.strip()) <= 60
                      or (strip(t).strip().isdigit() and 50 <= int(strip(t).strip()) <= 60)},
    {"id": "all_lowercase",
     "prompt": "Reply with the phrase 'system online' in all lowercase and nothing else.",
     "check": lambda t: strip(t).strip() == "system online"},
    {"id": "csv_three_fields",
     "prompt": "Output one line of three comma-separated values: red,green,blue (in that order).",
     "check": lambda t: strip(t).strip() == "red,green,blue"},
    {"id": "exact_word_count_5",
     "prompt": "Write a sentence about the moon using exactly five words.",
     "check": lambda t: _word_count(t) == 5},
]


def grade(case_id, response):
    case = next(c for c in CASES if c["id"] == case_id)
    try:
        return bool(case["check"](response))
    except Exception:
        return False
