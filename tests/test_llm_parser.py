"""Unit tests for core/llm_client.py: <think> stripping, Nano-Tag parsing, and broken JSON repair."""

from __future__ import annotations

from core.llm_client import extract_json, parse_nano_tags, strip_think


def test_strip_think_removes_reasoning_blocks_and_dangling_tags() -> None:
    raw = (
        "<think>\nLet me reason about quotes and newlines...\n</think>\n"
        "<<<FILE_PATH>>>\napp.py\n<<<END_FILE_PATH>>>\n"
        "<<<CODE>>>\nx = 1\n<<<END_CODE>>>"
    )
    cleaned = strip_think(raw)
    assert "<think>" not in cleaned
    assert "<<<FILE_PATH>>>" in cleaned

    unclosed = "<think>still thinking... <<<FILE_PATH>>>\nmod.py\n<<<END_FILE_PATH>>>"
    assert strip_think(unclosed).startswith("<<<FILE_PATH>>>")


def test_parse_nano_tags_handles_multiline_code_quotes_and_fences() -> None:
    response_text = """\
<think>Planning implementation of get_words()</think>
<<<FILE_PATH>>>
vocab/loader.py
<<<END_FILE_PATH>>>
<<<CODE>>>
```python
def get_words() -> list[dict]:
    \"\"\"Return words with quotes like "hello" and newline \\n escapes.\"\"\"
    text = "line1\\nline2"
    return [{"word": text}]
```
<<<END_CODE>>>
<<<RELAY_NOTE>>>
Implemented get_words() -> list[dict] inside vocab/loader.py.
Next agent should import get_words directly from vocab.loader.
<<<END_RELAY_NOTE>>>
"""
    parsed = parse_nano_tags(response_text)
    assert parsed.has_code
    assert len(parsed.files) == 1
    assert parsed.files[0].path == "vocab/loader.py"
    assert "def get_words() -> list[dict]:" in parsed.files[0].code
    assert "```" not in parsed.files[0].code
    assert "vocab.loader" in parsed.relay_note


def test_parse_nano_tags_tolerates_two_brackets_and_missing_end_tag() -> None:
    malformed = (
        "<<FILE_PATH>>\nutils/calc.py\n<<END_FILE_PATH>>\n"
        "<<CODE>>\ndef add(a: int, b: int) -> int:\n    return a + b\n"
        "<<RELAY_NOTE>>\nDefined add(a, b) -> int in utils/calc.py."
    )
    parsed = parse_nano_tags(malformed)
    assert parsed.has_code
    assert parsed.files[0].path == "utils/calc.py"
    assert "return a + b" in parsed.files[0].code
    assert "Defined add(a, b)" in parsed.relay_note


def test_extract_json_repairs_small_model_mistakes() -> None:
    broken_json = """\
<think>Decomposing into micro tasks</think>
Here is the plan:
```json
{
  // Architect output with single quotes, trailing commas, Python booleans, and missing closer
  micro_tasks: [
    {
      'id': 'step-1',
      'title': 'Create app.py',
      'target_file': 'app.py',
      'kind': 'code',
      'approved': True,
      'optional': None,
      'acceptance': ['Defines get_words()',],
    },
  ],
```
"""
    data = extract_json(broken_json, expect=dict)
    assert isinstance(data, dict)
    assert len(data["micro_tasks"]) == 1
    first = data["micro_tasks"][0]
    assert first["id"] == "step-1"
    assert first["target_file"] == "app.py"
    assert first["approved"] is True
    assert first["optional"] is None
