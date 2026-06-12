"""Multilingual quality checks: EN↔RU translation + RU reasoning."""
import re

# Translation pairs: (source, target_lang, reference_keywords)
TRANSLATION = [
    {"id": "en_ru_1", "prompt": "Translate to Russian: 'The server restarted after the update.'",
     "keywords": ["сервер", "перезапуст", "обновлен"]},
    {"id": "en_ru_2", "prompt": "Translate to Russian: 'Memory usage peaked at sixteen gigabytes.'",
     "keywords": ["памят", "шестнадцат", "гигабайт"]},
    {"id": "en_ru_3", "prompt": "Translate to Russian: 'Quantization reduces model size with minimal quality loss.'",
     "keywords": ["квант", "размер", "качеств"]},
    {"id": "ru_en_1", "prompt": "Translate to English: 'Кэш ключ-значение растёт линейно с длиной контекста.'",
     "keywords": ["cache", "context", "linear"]},
    {"id": "ru_en_2", "prompt": "Translate to English: 'Локальный инференс не требует отправки данных в облако.'",
     "keywords": ["local", "inference", "cloud"]},
]

# RU reasoning: keyword match against expected facts
RU_REASONING = [
    {"id": "ru_fact_1",
     "prompt": "Сколько бит в одном байте? Ответь одним числом.",
     "keywords": ["8"]},
    {"id": "ru_fact_2",
     "prompt": "Какой химический символ у водорода? Ответь одной буквой.",
     "keywords": ["H", "h"]},
    {"id": "ru_fact_3",
     "prompt": "Сколько будет 17 + 25? Ответь только числом.",
     "keywords": ["42"]},
    {"id": "ru_fact_4",
     "prompt": "Как называется процесс преобразования текста модели в последовательность токенов? "
               "Ответь одним словом на русском или английском.",
     "keywords": ["токениз", "tokeniz"]},
    {"id": "ru_fact_5",
     "prompt": "Если VRAM 16 ГБ, а веса модели ~15 ГБ, сколько гигабайт остаётся на KV-кэш? "
               "Ответь одним числом (целое, без единиц).",
     "keywords": ["1"]},
]

ALL = TRANSLATION + RU_REASONING


def _match_any(text, keywords):
    t = strip(text).lower()
    return any(k.lower() in t for k in keywords)


def grade_translation(response, keywords):
    return _match_any(response, keywords)


def grade_reasoning(response, keywords):
    stripped = strip(response).strip()
    for k in keywords:
        if stripped == k or stripped.lower() == k.lower():
            return True
    return _match_any(response, keywords)


def grade_case(case_id, response):
    case = next(c for c in ALL if c["id"] == case_id)
    if case_id.startswith(("en_ru", "ru_en")):
        return grade_translation(response, case["keywords"])
    return grade_reasoning(response, case["keywords"])
