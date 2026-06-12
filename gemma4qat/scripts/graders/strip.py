"""Strip Gemma 4 template artifacts before grading."""
import json
import re

THINKING = re.compile(r"<\|channel>thought\n.*?(?:<\|channel\|>|<channel\|>)", re.S)
FENCE = re.compile(r"```(?:json|python)?\s*\n(.*?)```", re.S)


def strip(text):
    if not text:
        return ""
    t = THINKING.sub("", text)
    t = t.replace("<|channel|>", "").replace("<channel|>", "")
    t = t.replace("<|turn>model\n", "").replace("<turn|>", "")
    return t.strip()


def extract_json(text):
    t = strip(text)
    m = FENCE.search(t)
    if m:
        t = m.group(1).strip()
    return t
