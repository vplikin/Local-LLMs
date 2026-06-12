# NIAH prompts (Gemma 4 QAT benchmark)

## Single-needle template
```
{haystack filler ~90% of num_ctx}

Important fact: the vault access code is ratatoskr-7741.

Question: What is the vault access code? Reply with the value only, nothing else.
```

## Multi-needle facts (depths 10%, 50%, 90%)
| key | value |
|-----|-------|
| vault access code | ratatoskr-7741 |
| relay callsign | anatr0p-zulu |
| checkpoint phrase | lord-yorkshire-9 |

## Depths tested
0, 10, 25, 50, 75, 90, 100 (% of haystack word count)

## Haystack filler
Neutral maritime navigation text (see `niah.py` FILLER constant). No digits or answer leakage.
