# HiveCoder

[![CI](https://github.com/VeyselGaziDuran/HiveCoder/actions/workflows/ci.yml/badge.svg)](https://github.com/VeyselGaziDuran/HiveCoder/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776ab.svg)](https://www.python.org/)

[English](#english) | [Türkçe](#türkçe)

---

## English

**HiveCoder** is a stateless micro-agent orchestrator for automated code generation and verification, running over local models (Ollama) or OpenAI/Anthropic-compatible APIs.

Instead of passing cumulative chat history across steps, HiveCoder decomposes a task into single-file atomic units, indexes public module interfaces with Python's `ast` module onto a shared **Blackboard**, and verifies each step with `py_compile` and `pytest` before advancing.

### Core Logic

1. **Task Decomposition & Environment Probes (`ArchitectAgent` & `probes/`):** Inspects active Python/package versions (`RuntimeProbe`, `PackageRadarProbe`) and breaks a high-level specification into an ordered queue of single-file micro-tasks (`MicroTask`) ordered by dependency.
2. **AST-Indexed Shared Memory (`core/blackboard.py`):** Each agent runs in an isolated session with no conversation history. Whenever a file is written, `ast.parse()` extracts class definitions, method/function signatures (`def func(a: int) -> list[dict]`), module constants, and imports into `memory/blackboard.json` alongside a short hand-off note. Downstream agents receive only this interface map and the target file (`core/context_builder.py`).
3. **Surgical AST Merging & Block-Tag Protocol (`core/ast_patcher.py` & `core/llm_client.py`):** Code-producing agents emit deterministic block delimiters (`<<<FILE_PATH>>>`, `<<<CODE>>>`, `<<<RELAY_NOTE>>>`). When updating an existing module, `ASTPatcher` preserves previously verified functions and classes so partial outputs from smaller models cannot overwrite working symbols.
4. **Static Inspection & Automated Testing (`InspectorAgent` & `Validator`):** Each generated file is checked for AST syntax validity and cross-module import consistency against the Blackboard, followed by `py_compile` and `pytest` execution with automatic `PYTHONPATH` injection.
5. **Isolated Fault Repair (`MedicAgent`), Rollback (`GitGuard`) & Ephemeral Lifecycle (`MemoryLifecycle`):** If inspection or test execution fails, `MedicAgent` receives the failing file, extracted traceback line windows, real callable signatures (`SignatureProbe`), and the Blackboard summary. Unrecoverable steps are rolled back to their pre-step snapshot and their temporary notes are purged. Upon full task completion, all transient notes, snapshots, and caches are flushed automatically.

### Installation

#### Prerequisites
- Python `3.10+`
- Git
- [Ollama](https://ollama.com/) for local models, or an OpenAI / Anthropic-compatible API key

```bash
git clone https://github.com/VeyselGaziDuran/HiveCoder.git
cd HiveCoder

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
venv/bin/pip install -r requirements.txt

# Optional: install the `hive` CLI entry point
venv/bin/pip install -e .
```

If using a local model via Ollama:

```bash
ollama serve &
ollama pull qwen3:8b
```

### Usage

#### 1. Interactive Shell (REPL)

```bash
venv/bin/python hive.py
```

Enter a task directly or use slash commands inside the shell:

```text
hive> /workspace ./my_project
hive> /model qwen3:8b
hive> /run Create a SQLite-backed key-value store module with pytest unit tests
hive> /blackboard
hive> /status
hive> /exit
```

#### 2. Command Line (CLI)

```bash
# Run a single autonomous task in a target directory
venv/bin/python hive.py \
  --task "Build a JSON vocabulary module and its pytest suite" \
  --workspace ./vocab_app

# Resume an interrupted task from memory/state.json
venv/bin/python hive.py --resume

# Run against an OpenAI / Anthropic / OpenRouter / DeepSeek API
export HIVE_API_KEY="sk-..."
venv/bin/python hive.py \
  --provider api \
  --model gpt-4o-mini \
  --task "Implement a token bucket rate limiter with unit tests" \
  --workspace ./rate_limiter
```

#### Command Reference

| Command | Description |
| :--- | :--- |
| `/run <goal>` | Start a new task. |
| `/resume` | Resume an interrupted task from `memory/state.json`. |
| `/blackboard` | Display indexed AST signatures and agent relay notes. |
| `/provider <ollama\|api>` | Switch between local Ollama and remote API providers. |
| `/model <model_name>` | Change the active model and persist it to `config.json`. |
| `/workspace [path]` | Show or change the target workspace directory. |
| `/status` | Display active task progress, step statuses, and metrics. |
| `/clean` | Remove temporary cache directories (`__pycache__`, `.pytest_cache`, `.hive_tmp`). |
| `/self-test` | Run the internal unit and integration test suite. |
| `/help` | Show available commands. |
| `/exit` | Exit the session. |

### Configuration (`config.json`)

```json
{
  "provider": "ollama",
  "ollama": {
    "base_url": "http://localhost:11434",
    "model": "qwen3:8b",
    "num_ctx": 8192,
    "think": false
  },
  "api": {
    "style": "openai",
    "base_url": "https://api.openai.com/v1",
    "api_key_env": "HIVE_API_KEY",
    "model": "gpt-4o-mini",
    "max_tokens": 4096
  },
  "generation": {
    "temperature": 0.2,
    "timeout_seconds": 600,
    "max_retries": 3
  },
  "limits": {
    "max_micro_tasks": 30,
    "max_medic_attempts": 3,
    "max_coder_attempts": 2,
    "max_context_chars": 14000,
    "max_relay_notes_in_context": 12,
    "pytest_timeout_seconds": 180
  }
}
```

- **OpenAI-compatible endpoints (OpenRouter, DeepSeek, Groq, etc.):** Set `api.base_url` and `api.model`, and export your key via `HIVE_API_KEY`.
- **Anthropic Claude API:** Set `api.style` to `"anthropic"` and `api.base_url` to `"https://api.anthropic.com/v1"`.

---

## Türkçe

**HiveCoder**, yerel modeller (Ollama) ve OpenAI/Anthropic uyumlu API'ler üzerinde çalışan, durumsuz (stateless) bir mikro-ajan kod üretim ve doğrulama orkestratörüdür.

Ajanlar arasında sohbet geçmişi taşımak yerine her adımı tek dosyalık atomik görevlere böler, üretilen dosyaların sınıf ve fonksiyon imzalarını Python `ast` modülüyle ortak bir **Blackboard** üzerine indeksler ve her adımı `py_compile` + `pytest` ile doğrulayarak ilerler.

### Çalışma Mantığı

1. **Görev Ayrıştırma ve Ortam Sondaları (`ArchitectAgent` & `probes/`):** Aktif Python ve paket sürümlerini (`RuntimeProbe`, `PackageRadarProbe`) tespit eder ve hedefi bağımlılık sırasına göre tek dosyalık mikro-görev kuyruğuna (`MicroTask`) dönüştürür.
2. **AST Tabanlı Ortak Hafıza (`core/blackboard.py`):** Her ajan sıfır sohbet geçmişiyle çalışır. Bir dosya yazıldığında `ast.parse()` ile içindeki sınıf, metot/fonksiyon imzaları (`def func(a: int) -> list[dict]`), sabitler ve importlar çıkarılarak `memory/blackboard.json` içine kaydedilir. Sıradaki ajan tüm projeyi okumak yerine yalnızca bu arayüz haritasını ve hedef dosyayı alır (`core/context_builder.py`).
3. **Cerrahi AST Birleştirme ve Blok Etiket Protokolü (`core/ast_patcher.py` & `core/llm_client.py`):** Kod üreten ajanlar deterministik blok etiketleri (`<<<FILE_PATH>>>`, `<<<CODE>>>`, `<<<RELAY_NOTE>>>`) kullanır. Mevcut bir dosya güncellenirken `ASTPatcher`, daha önce doğrulanmış sınıf ve fonksiyonları koruyarak küçük modellerin çalışan kodları silmesini engeller.
4. **Statik Denetim ve Otomatik Test (`InspectorAgent` & `Validator`):** Üretilen her dosya önce AST sözdizimi ve modüller arası import tutarlılığı açısından denetlenir, ardından `PYTHONPATH` otomatik enjekte edilerek `py_compile` ve `pytest` çalıştırılır.
5. **İzole Hata Onarımı (`MedicAgent`), Geri Sarma (`GitGuard`) ve Geçici Hafıza Temizliği (`MemoryLifecycle`):** Doğrulamadan geçemeyen dosyalar yalnızca hata satırı penceresi, gerçek fonksiyon imzaları (`SignatureProbe`) ve Blackboard arayüz özetiyle onarım döngüsüne alınır. Onarılamayan adımlar görev öncesi snapshot'a geri sarılır ve geçici notları silinir. Görev tamamlandığında tüm geçici devir notları, snapshot'lar ve önbellekler otomatik olarak temizlenir.

### Kurulum

#### Gereksinimler
- Python `3.10+`
- Git
- Yerel kullanım için [Ollama](https://ollama.com/) veya uzak API kullanımı için OpenAI / Anthropic uyumlu bir API anahtarı

```bash
git clone https://github.com/VeyselGaziDuran/HiveCoder.git
cd HiveCoder

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
venv/bin/pip install -r requirements.txt

# İsteğe bağlı: 'hive' komutunu doğrudan kullanmak için
venv/bin/pip install -e .
```

Yerel model kullanacaksanız Ollama servisini başlatıp bir model indirin:

```bash
ollama serve &
ollama pull qwen3:8b
```

### Kullanım

#### 1. İnteraktif Kabuk (REPL)

```bash
venv/bin/python hive.py
```

Kabuk içinde doğrudan görev yazabilir veya slash komutlarını kullanabilirsiniz:

```text
hive> /workspace ./my_project
hive> /model qwen3:8b
hive> /run SQLite tabanlı anahtar-değer (key-value) saklama modülü ve pytest birim testlerini yaz
hive> /blackboard
hive> /status
hive> /exit
```

#### 2. Komut Satırı (CLI)

```bash
# Tek satırlık görev çalıştırma
venv/bin/python hive.py \
  --task "Build a JSON vocabulary module and its pytest suite" \
  --workspace ./vocab_app

# Yarıda kalan göreve kaldığı adımdan devam etme
venv/bin/python hive.py --resume

# Uzak API (OpenAI / Anthropic / OpenRouter / DeepSeek) ile çalıştırma
export HIVE_API_KEY="sk-..."
venv/bin/python hive.py \
  --provider api \
  --model gpt-4o-mini \
  --task "Implement a token bucket rate limiter with unit tests" \
  --workspace ./rate_limiter
```

#### Komut Referansı

| Komut | Açıklama |
| :--- | :--- |
| `/run <görev>` | Yeni bir görev başlatır. |
| `/resume` | Kesintiye uğrayan görevi `memory/state.json` üzerinden devam ettirir. |
| `/blackboard` | İndekslenen AST imzalarını ve ajan devir notlarını gösterir. |
| `/provider <ollama\|api>` | Yerel Ollama veya uzak API sağlayıcısı arasında geçiş yapar. |
| `/model <model_adı>` | Aktif modeli değiştirir ve `config.json` içine kaydeder. |
| `/workspace [yol]` | Hedef çalışma dizinini gösterir veya değiştirir. |
| `/status` | Aktif görev durumunu, tamamlanan adımları ve metrikleri gösterir. |
| `/clean` | Geçici önbellek dosyalarını (`__pycache__`, `.pytest_cache`, `.hive_tmp`) temizler. |
| `/self-test` | Proje birim ve entegrasyon testlerini çalıştırır. |
| `/help` | Komut listesini gösterir. |
| `/exit` | Oturumu sonlandırır. |

### Yapılandırma (`config.json`)

- **OpenAI uyumlu servisler (OpenRouter, DeepSeek, Groq vb.):** `config.json` içindeki `api.base_url` ve `api.model` alanlarını güncelleyip API anahtarını `HIVE_API_KEY` ortam değişkenine tanımlayın.
- **Anthropic Claude API:** `api.style` değerini `"anthropic"` ve `api.base_url` değerini `"https://api.anthropic.com/v1"` olarak ayarlayın.

---

## Project Structure / Proje Yapısı

```text
HiveCoder/
├── hive.py                  # CLI & interactive TUI entry point
├── config.json              # Runtime configuration
├── core/
│   ├── orchestrator.py      # Task execution & verification pipeline
│   ├── blackboard.py        # AST interface extraction & relay notes
│   ├── ast_patcher.py       # Surgical AST merging & traceback window extractor
│   ├── memory_lifecycle.py  # Ephemeral note & snapshot lifecycle manager
│   ├── context_builder.py   # Target file + Blackboard prompt builder
│   ├── llm_client.py        # Ollama / API client & block-tag parser
│   ├── state_machine.py     # State transitions & crash-safe resume
│   ├── validator.py         # py_compile & pytest runner with PYTHONPATH injection
│   ├── git_guard.py         # Isolated snapshot & rollback manager
│   └── garbage_collector.py # Temporary file & cache cleanup
├── probes/
│   ├── runtime_probe.py     # Active Python & installed package inspector
│   ├── package_radar.py     # Live PyPI version lookup with offline fallback
│   └── signature_probe.py   # Real callable signature & docstring extractor
├── agents/
│   ├── architect_agent.py   # Task decomposition agent
│   ├── coder_agent.py       # Module implementation agent
│   ├── inspector_agent.py   # Static AST & code review agent
│   ├── tester_agent.py      # Pytest generation agent
│   └── medic_agent.py       # Traceback-driven repair agent
├── ui/
│   └── terminal_ui.py       # Rich terminal interface
└── tests/                   # Unit & end-to-end integration tests
```

## Tests / Testler

```bash
venv/bin/pytest -v
```

## License / Lisans

MIT License — Copyright (c) 2026 [Veysel Gazi Duran](https://github.com/VeyselGaziDuran). See [LICENSE](LICENSE).
