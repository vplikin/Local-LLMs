import re

from .strip import strip

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _norm(s):
    if s is None:
        return None
    return s.replace(",", "").rstrip(".")


def extract_final(text):
    nums = NUM.findall(strip(text))
    return _norm(nums[-1]) if nums else None


def gold(answer_field):
    m = re.search(r"####\s*(-?[\d,]*\.?\d+)", answer_field)
    return _norm(m.group(1)) if m else None


def grade(pred_text, answer_field):
    p, g = extract_final(pred_text), gold(answer_field)
    if p is None or g is None:
        return False
    try:
        return float(p) == float(g)
    except ValueError:
        return p == g
