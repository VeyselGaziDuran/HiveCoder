# Contributing

Contributions, bug reports, and pull requests are welcome.

## Design Invariants

1. **Stateless Agent Invocations:** Agents under `agents/` must not retain conversation history between calls. Cross-step state belongs on the `Blackboard` (`core/blackboard.py`).
2. **Block-Tag Code Output:** Code-generating agents (`CoderAgent`, `TesterAgent`, `MedicAgent`) emit source files via `<<<FILE_PATH>>>`, `<<<CODE>>>`, and `<<<RELAY_NOTE>>>` blocks rather than JSON strings.
3. **Verified Completion:** `StateMachine.complete_micro_task()` requires written files, `InspectorAgent` approval, and a passing `Validator` run (`py_compile` + `pytest`).
4. **No Unfinished Code:** All submitted modules and tests must be complete and type-annotated.

## Local Development

```bash
git clone https://github.com/VeyselGaziDuran/HiveCoder.git
cd HiveCoder
python3 -m venv venv
source venv/bin/activate
venv/bin/pip install -r requirements.txt
```

## Running the Test Suite

```bash
venv/bin/pytest -v
# or
venv/bin/python hive.py --self-test
```

Ensure all tests pass and no cache artifacts remain in the working tree before submitting a pull request.
