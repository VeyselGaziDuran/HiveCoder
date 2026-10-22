"""Hybrid LLM client (Ollama / OpenAI-compatible / Anthropic) + bullet-proof parsers.

Small local models (3B-8B) routinely break JSON when they have to embed
multi-line source code (unescaped quotes, raw newlines ...).  HiveCoder never
asks for code inside JSON.  Code travels inside *Nano-Tags*::

    <<<FILE_PATH>>>
    pkg/module.py
    <<<END_FILE_PATH>>>
    <<<CODE>>>
    ...pure source...
    <<<END_CODE>>>
    <<<RELAY_NOTE>>>
    Two sentences for the next agent.
    <<<END_RELAY_NOTE>>>

:func:`parse_nano_tags` tolerates missing end tags, two or three angle
brackets, ``<<</CODE>>>`` closers, markdown fences inside blocks and leading
``<think>`` reasoning.  :func:`extract_json` repairs the usual small-model JSON
damage (trailing commas, single quotes, Python literals, comments, unquoted
keys, raw newlines, missing closers).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

# ---------------------------------------------------------------------- think blocks
_THINK_BLOCK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN = re.compile(r"<(think|thinking|reasoning)>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</(think|thinking|reasoning)>", re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning, including unbalanced fragments."""
    if not text:
        return ""
    cleaned = _THINK_BLOCK.sub("", text)
    closes = list(_THINK_CLOSE.finditer(cleaned))
    if closes:
        cleaned = cleaned[closes[-1].end():]
    opened = _THINK_OPEN.search(cleaned)
    if opened:
        tail = cleaned[opened.end():]
        anchor = re.search(r"<{2,3}\s*(FILE_PATH|CODE|RELAY_NOTE)|```|[{\[]", tail, re.IGNORECASE)
        cleaned = cleaned[: opened.start()] + (tail[anchor.start():] if anchor else "")
    return cleaned.strip()


# ---------------------------------------------------------------------- nano tags
_TAG = re.compile(r"<{2,3}\s*(/|END[_ ]?)?\s*([A-Za-z][A-Za-z_ ]*?)\s*>{2,3}", re.IGNORECASE)
_FENCE_BLOCK = re.compile(r"```[ \t]*([\w+-]*)[ \t]*\n(.*?)```", re.DOTALL)
OPENING_TAGS: frozenset[str] = frozenset({"FILE_PATH", "CODE", "RELAY_NOTE"})


class LLMParseError(ValueError):
    """Raised when a model response cannot be turned into structured data."""


class LLMError(RuntimeError):
    """Raised when the model backend cannot be reached or keeps failing."""


@dataclass
class FileBlock:
    path: Optional[str]
    code: str


@dataclass
class NanoResponse:
    files: list[FileBlock] = field(default_factory=list)
    relay_note: str = ""
    tags: dict[str, list[str]] = field(default_factory=dict)
    raw: str = ""

    @property
    def has_code(self) -> bool:
        return any(block.code.strip() for block in self.files)


def strip_code_fences(code: str) -> str:
    """Remove a surrounding markdown fence (```python ... ```) if present."""
    stripped = code.strip("\n")
    lines = stripped.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
    elif lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def _clean_path(raw: str) -> str:
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        line = re.sub(r"^[#*>-]+\s*", "", line).strip().strip("`'\"*").strip()
        line = re.sub(
            r"^(?:file_?path|target_?file|file_?name|path|file)\s*:\s*",
            "",
            line,
            flags=re.IGNORECASE,
        ).strip().strip("`'\"*").strip()
        if line and not line.startswith("```"):
            return line
    return ""


_STRAY_TAG_LINE = re.compile(
    r"^\s*<{2,3}\s*(?:/|END(?:[_ ]|\b)|FILE_PATH\b|CODE\b|RELAY_NOTE\b).*?>{2,3}\s*$",
    re.IGNORECASE,
)
_JSON_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _normalize_code(code: str) -> str:
    code = strip_code_fences(code)
    lines = code.splitlines()
    while lines and (not lines[0].strip() or _STRAY_TAG_LINE.match(lines[0])):
        lines.pop(0)
    while lines and (not lines[-1].strip() or _STRAY_TAG_LINE.match(lines[-1])):
        lines.pop()
    return ("\n".join(lines) + "\n") if lines else ""


def _split_tag_blocks(text: str) -> list[tuple[str, str]]:
    """Return ``[(TAG_NAME, content), ...]`` in order of appearance."""
    markers = []
    for match in _TAG.finditer(text):
        name = re.sub(r"\s+", "_", match.group(2).strip().upper())
        is_end = bool(match.group(1))
        if name.startswith("END_"):
            name, is_end = name[4:], True
        markers.append((match.start(), match.end(), name, is_end))

    blocks: list[tuple[str, str]] = []
    i = 0
    while i < len(markers):
        _, end, name, is_end = markers[i]
        if is_end or name not in OPENING_TAGS:
            i += 1
            continue
        content_end = len(text)
        j = i + 1
        while j < len(markers):
            m_start, _, m_name, m_end = markers[j]
            if m_end and m_name == name:
                content_end = m_start
                j += 1
                break
            # An opening tag ends the current block, except CODE inside CODE
            # (source code may legitimately mention the word "CODE").
            if not m_end and m_name in OPENING_TAGS and not (name == "CODE" and m_name == "CODE"):
                content_end = m_start
                break
            j += 1
        blocks.append((name, text[end:content_end]))
        i = j
    return blocks


def parse_nano_tags(text: str) -> NanoResponse:
    """Parse a Nano-Tag response into files + relay note."""
    cleaned = strip_think(text)
    response = NanoResponse(raw=text)
    pending_path: Optional[str] = None
    for name, content in _split_tag_blocks(cleaned):
        response.tags.setdefault(name, []).append(content.strip())
        if name == "FILE_PATH":
            pending_path = _clean_path(content) or None
        elif name == "CODE":
            code = _normalize_code(content)
            if code:
                response.files.append(FileBlock(path=pending_path, code=code))
            pending_path = None
        elif name == "RELAY_NOTE":
            note = re.sub(r"\s+", " ", strip_code_fences(content)).strip()
            if note:
                response.relay_note = (response.relay_note + " " + note).strip()

    if not response.files:
        fences = [(lang, body) for lang, body in _FENCE_BLOCK.findall(cleaned) if body.strip()]
        if fences:
            code_langs = {
                "python", "py", "typescript", "ts", "tsx", "javascript", "js", "jsx",
                "dart", "kotlin", "kt", "swift", "json", "html", "css", "scss",
                "sql", "sh", "bash", "yaml", "yml", "toml",
            }
            preferred = [f for f in fences if f[0].lower() in code_langs] or [
                f for f in fences if f[0] == ""
            ] or fences
            body = max(preferred, key=lambda f: len(f[1]))[1]
            path = response.tags.get("FILE_PATH", [None])[0]
            response.files.append(FileBlock(path=_clean_path(path) if path else None, code=_normalize_code(body)))
    return response


# ---------------------------------------------------------------------- JSON
def _repair_json(src: str) -> str:
    """Character-level repair of common LLM JSON mistakes."""
    out: list[str] = []
    stack: list[str] = []
    i, n = 0, len(src)
    pairs = {"{": "}", "[": "]"}

    def drop_trailing_comma() -> None:
        j = len(out) - 1
        while j >= 0 and out[j].isspace():
            j -= 1
        if j >= 0 and out[j] == ",":
            del out[j]

    while i < n:
        ch = src[i]
        if ch in "\"'":
            quote = ch
            i += 1
            buf = ['"']
            closed = False
            while i < n:
                c = src[i]
                if c == "\\" and i + 1 < n:
                    nxt = src[i + 1]
                    if nxt == "'":
                        buf.append("'")
                    elif nxt in '"\\/bfnrtu':
                        buf.append(c + nxt)
                    else:
                        buf.append("\\\\" + nxt)
                    i += 2
                    continue
                if c == quote:
                    i += 1
                    closed = True
                    break
                if c == '"':
                    buf.append('\\"')
                elif c == "\n":
                    buf.append("\\n")
                elif c == "\r":
                    buf.append("\\r")
                elif c == "\t":
                    buf.append("\\t")
                else:
                    buf.append(c)
                i += 1
            buf.append('"')
            out.append("".join(buf))
            if not closed:
                break
            continue
        if ch == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and src[i + 1] == "*":
            end = src.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        if ch == "#":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if ch in "{[":
            stack.append(pairs[ch])
            out.append(ch)
            i += 1
            continue
        if ch in "}]":
            drop_trailing_comma()
            if stack and stack[-1] == ch:
                stack.pop()
                out.append(ch)
            i += 1
            if not stack:
                break
            continue
        if ch.isdigit() or (ch == "-" and i + 1 < n and src[i + 1].isdigit()):
            num_match = _JSON_NUMBER.match(src, i)
            if num_match:
                out.append(num_match.group(0))
                i = num_match.end()
                continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (src[j].isalnum() or src[j] in "_-$"):
                j += 1
            word = src[i:j]
            k = j
            while k < n and src[k] in " \t":
                k += 1
            if k < n and src[k] == ":":
                out.append(json.dumps(word))
            elif word in ("True", "true"):
                out.append("true")
            elif word in ("False", "false"):
                out.append("false")
            elif word in ("None", "none", "null", "Null", "NULL", "NaN", "undefined"):
                out.append("null")
            else:
                out.append(json.dumps(word))
            i = j
            continue
        out.append(ch)
        i += 1

    drop_trailing_comma()
    while stack:
        out.append(stack.pop())
    return "".join(out)


def _balanced_candidates(text: str) -> list[str]:
    """Substrings that start at ``{``/``[`` and end at the matching closer (or EOF)."""
    candidates: list[str] = []
    starts = [m.start() for m in re.finditer(r"[{\[]", text)][:24]
    for start in starts:
        depth = 0
        in_str: Optional[str] = None
        escape = False
        end = None
        for idx in range(start, len(text)):
            c = text[idx]
            if in_str:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == in_str:
                    in_str = None
                continue
            if c in "\"'":
                in_str = c
            elif c in "{[":
                depth += 1
            elif c in "}]":
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break
        candidate = text[start:end] if end else text[start:]
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def extract_json(text: str, expect: Optional[type] = None) -> Any:
    """Extract (and if needed repair) the first JSON value in an LLM response."""
    cleaned = strip_think(text).strip()
    if not cleaned:
        raise LLMParseError("Empty response")

    def accept(value: Any) -> bool:
        return expect is None or isinstance(value, expect)

    attempts: list[str] = [cleaned]
    attempts.extend(body for _, body in _FENCE_BLOCK.findall(cleaned))
    attempts.extend(_balanced_candidates(cleaned))
    for candidate in attempts:
        candidate = candidate.strip()
        if not candidate:
            continue
        for variant in (candidate, _repair_json(candidate)):
            try:
                value = json.loads(variant)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(value, (dict, list)) and accept(value):
                return value
    raise LLMParseError(f"No valid JSON object found in response: {cleaned[:200]!r}")


# ---------------------------------------------------------------------- client
def unload_ollama_model(
    base_url: str,
    model: str,
    session: Optional[requests.Session] = None,
    timeout: float = 15.0,
) -> bool:
    """Ask Ollama to evict ``model`` from RAM/VRAM immediately (``keep_alive: 0``)."""
    http = session or requests
    try:
        resp = http.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=timeout,
        )
    except requests.RequestException:
        return False
    return 200 <= resp.status_code < 300


@dataclass
class ClientStats:
    calls: int = 0
    prompt_chars: int = 0
    response_chars: int = 0
    busy_seconds: float = 0.0
    errors: int = 0


class LLMClient:
    """Stateless completion client: every call is a brand-new conversation."""

    def __init__(self, config: dict[str, Any], session: Optional[requests.Session] = None) -> None:
        self.config = config
        self.session = session or requests.Session()
        self.stats = ClientStats()

    @property
    def provider(self) -> str:
        return str(self.config.get("provider", "ollama"))

    @property
    def model(self) -> str:
        section = self.config.get("ollama" if self.provider == "ollama" else "api", {})
        return str(section.get("model", ""))

    @property
    def _gen(self) -> dict[str, Any]:
        return self.config.get("generation", {})

    def complete(self, system: str, user: str, temperature: Optional[float] = None) -> str:
        retries = max(1, int(self._gen.get("max_retries", 3)))
        temp = float(self._gen.get("temperature", 0.2) if temperature is None else temperature)
        last_error: Optional[Exception] = None
        for attempt in range(retries):
            started = time.time()
            try:
                if self.provider == "ollama":
                    text = self._ollama(system, user, temp)
                elif self.config.get("api", {}).get("style", "openai") == "anthropic":
                    text = self._anthropic(system, user, temp)
                else:
                    text = self._openai(system, user, temp)
                self.stats.calls += 1
                self.stats.prompt_chars += len(system) + len(user)
                self.stats.response_chars += len(text)
                return text
            except (requests.RequestException, LLMError, KeyError, ValueError) as exc:
                last_error = exc
                self.stats.errors += 1
                if attempt < retries - 1:
                    time.sleep(min(2 ** attempt, 8))
            finally:
                self.stats.busy_seconds += time.time() - started
        raise LLMError(f"{self.provider} request failed after {retries} attempts: {last_error}")

    def _post(self, url: str, payload: dict[str, Any], headers: Optional[dict[str, str]] = None) -> dict[str, Any]:
        timeout = float(self._gen.get("timeout_seconds", 600))
        try:
            resp = self.session.post(url, json=payload, headers=headers or {}, timeout=timeout)
        except requests.ConnectionError as exc:
            raise LLMError(f"Cannot reach {url}. Is the server running? ({exc})") from exc
        if resp.status_code >= 400:
            raise LLMError(f"HTTP {resp.status_code} from {url}: {resp.text[:300]}")
        return resp.json()

    def _ollama(self, system: str, user: str, temperature: float) -> str:
        cfg = self.config.get("ollama", {})
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {
                "temperature": temperature,
                "num_ctx": int(cfg.get("num_ctx", 8192)),
            },
        }
        if cfg.get("think") is not None:
            payload["think"] = bool(cfg.get("think"))
        data = self._post(f"{cfg.get('base_url', 'http://localhost:11434').rstrip('/')}/api/chat", payload)
        return str(data["message"]["content"])

    def _api_key(self) -> str:
        cfg = self.config.get("api", {})
        key = os.environ.get(str(cfg.get("api_key_env", "HIVE_API_KEY")), "") or str(cfg.get("api_key", ""))
        if not key:
            raise LLMError(f"API key missing: export {cfg.get('api_key_env', 'HIVE_API_KEY')}=...")
        return key

    def _openai(self, system: str, user: str, temperature: float) -> str:
        cfg = self.config.get("api", {})
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": int(cfg.get("max_tokens", 4096)),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        headers = {"Authorization": f"Bearer {self._api_key()}"}
        data = self._post(f"{str(cfg.get('base_url')).rstrip('/')}/chat/completions", payload, headers)
        return str(data["choices"][0]["message"]["content"] or "")

    def _anthropic(self, system: str, user: str, temperature: float) -> str:
        cfg = self.config.get("api", {})
        payload = {
            "model": self.model,
            "max_tokens": int(cfg.get("max_tokens", 4096)),
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {"x-api-key": self._api_key(), "anthropic-version": "2023-06-01"}
        data = self._post(f"{str(cfg.get('base_url')).rstrip('/')}/messages", payload, headers)
        return "".join(part.get("text", "") for part in data.get("content", []) if part.get("type") == "text")

    def unload(self) -> bool:
        """Evict the current model from memory (only meaningful for Ollama)."""
        if self.provider != "ollama":
            return False
        base = self.config.get("ollama", {}).get("base_url", "http://localhost:11434")
        return unload_ollama_model(base, self.model, session=self.session)

    def health(self) -> tuple[bool, str]:
        """Quick reachability probe used by the TUI banner."""
        if self.provider != "ollama":
            env_name = str(self.config.get("api", {}).get("api_key_env", "HIVE_API_KEY"))
            present = bool(os.environ.get(env_name) or self.config.get("api", {}).get("api_key"))
            return present, ("api key present" if present else f"api key missing (export {env_name})")
        base = self.config.get("ollama", {}).get("base_url", "http://localhost:11434").rstrip("/")
        try:
            resp = self.session.get(f"{base}/api/tags", timeout=3)
            if resp.status_code >= 400:
                return False, f"Ollama HTTP {resp.status_code} at {base}"
            payload = resp.json()
            models = [m.get("name", "") for m in (payload.get("models", []) if isinstance(payload, dict) else [])]
        except (requests.RequestException, ValueError):
            return False, f"Ollama not reachable at {base}"
        if self.model not in models and f"{self.model}:latest" not in models:
            return False, f"model '{self.model}' not pulled (ollama pull {self.model})"
        return True, f"{len(models)} local models"
