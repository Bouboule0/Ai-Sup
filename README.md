# Ai-Sup

A minimal, readable, and testable orchestrator based on the **Plan-then-Execute (P-t-E)** paradigm, supporting **multi-model and cross-provider AI pairings** (Google Gemini & Anthropic Claude):

- **Frontier Planner (`BasePlanner`)**: Designs an atomic, Pydantic-validated execution plan (`Plan`) without writing final application code (`Gemini 4 Argon`, `Claude Opus 5.5`).
- **Cost-Efficient Executor (`BaseExecutor`)**: Executes each atomic step (code generation, file edits, tool calls) using a short ReAct loop (`Gemini 3.8 Flash`, `Gemini 3.1 Pro`, `Claude Sonnet 5.5`, `Claude Haiku 4.5`).
- **StepRouter & CostTracker**: Dynamic routing with conditional escalation back to the Planner upon repeated failures, and strict real-time budget enforcement.

Repository: https://github.com/Bouboule0/Ai-Sup

---

## Why Plan-then-Execute? Frontier-Only vs. Delegated Orchestration

In autonomous software engineering, token consumption is heavily skewed:
- **Architecture & Planning (~5–10% of tokens)** requires deep frontier reasoning, global context awareness, and dependency analysis.
- **Step Execution (~90–95% of tokens)** consists of repetitive ReAct loops: reading existing files, generating implementation code for a well-scoped task, running tests, and inspecting stdout/stderr.

Using a frontier model for every single file read, test run, and syntax fix quickly exhausts a student or indie developer's budget. **Ai-Sup** separates architectural reasoning from atomic execution.

### Result Quality, Speed & Behavior Comparison

| Dimension | 100% Frontier Model (`Argon` or `Opus` only) | Delegated P-t-E (`Planner` → `Executor`) |
| :--- | :--- | :--- |
| **Global Architecture** | Excellent — maintains high-level vision throughout. | **Identical** — the Frontier Planner still designs the full `Plan` and `Step` contracts. |
| **Atomic Code Generation** | Very high, even on ambiguous instructions. | **High when steps are atomic** — `Flash` / `Sonnet` / `3.1 Pro` excel when given clear `success_criteria`. |
| **Execution Speed (Latency)** | Slower — high time-to-first-token on every tool call. | **2× to 2.7× faster** — lightweight Executors iterate rapidly through file reads and test runs. |
| **Failure Recovery** | Self-corrects inline, but at frontier token prices (\$10–\$25/1M out). | **Smart Escalation** — Executor retries locally; on repeated failure (`>= 2`), `StepRouter` escalates back to the Planner to replan. |

---

## Comparative Multi-Model Benchmark (10-Step Pipeline)

Reproduce this table anytime via `orch benchmark` (or export it with `orch benchmark --output-md benchmark_results.md`).

The benchmark runs a 10-step software engineering pipeline (file inspection, Pydantic modeling, parser implementation, cyclic dependency graph solver, CLI generation, unit tests, and `pytest` verification) through `CostTracker`, `StepRouter`, and `ExecutionContext`:

| Pairing ID | Architecture | Planner (Parent) | Executor (Child) | Completion | ReAct Turns | Escalations | Planner Cost | Executor Cost | Total Cost | Savings vs Baseline | Latency |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `argon-only` | 100% Frontier | `gemini-4-argon` | `gemini-4-argon` | 10/10 (100%) | 20 | 0 | \$0.1290 | \$3.0200 | **\$3.1490** | Baseline (0.0%) | 152.5s |
| `argon-flash` | Delegated P-t-E | `gemini-4-argon` | `gemini-3.8-flash` | 10/10 (100%) | 24 | 1 | \$0.2580 | \$1.2420 | **\$1.5000** | **-52.4%** | 62.6s |
| `argon-pro` | Delegated P-t-E | `gemini-4-argon` | `gemini-3.1-pro` | 10/10 (100%) | 20 | 0 | \$0.1290 | \$1.6000 | **\$1.7290** | **-45.1%** | 90.5s |
| `opus-only` | 100% Frontier | `claude-opus-5.5` | `claude-opus-5.5` | 10/10 (100%) | 20 | 0 | \$0.3450 | \$8.0000 | **\$8.3450** | Baseline (0.0%) | 177.8s |
| `opus-sonnet` | Delegated P-t-E | `claude-opus-5.5` | `claude-sonnet-5.5` | 10/10 (100%) | 20 | 0 | \$0.3450 | \$2.1600 | **\$2.5050** | **-70.0%** | 77.8s |
| `opus-haiku` | Delegated P-t-E | `claude-opus-5.5` | `claude-haiku-4.5` | 10/10 (100%) | 36 | 1 | \$0.6900 | \$0.9576 | **\$1.6476** | **-80.3%** | 73.6s |
| `opus-flash` | Delegated P-t-E | `claude-opus-5.5` | `gemini-3.8-flash` | 10/10 (100%) | 24 | 1 | \$0.6900 | \$1.2420 | **\$1.9320** | **-76.8%** | 65.2s |
| `argon-sonnet` | Delegated P-t-E | `gemini-4-argon` | `claude-sonnet-5.5` | 10/10 (100%) | 20 | 0 | \$0.1290 | \$2.1600 | **\$2.2890** | **-27.3%** | 76.5s |

### Key Takeaways from the Benchmark
1. **`argon-flash` (`Gemini 4 Argon` → `Gemini 3.8 Flash`)**: Cuts cost by **52.4%** (\$1.50 vs \$3.15) and reduces latency by **2.4×** (62.6s vs 152.5s), even when accounting for 1 automatic escalation back to Argon on the complex cyclic graph solver step (`step_4`).
2. **`argon-pro` (`Gemini 4 Argon` → `Gemini 3.1 Pro`)**: Zero escalations needed on complex algorithmic steps while still saving **45.1%** (\$1.73 vs \$3.15).
3. **`opus-sonnet` (`Claude Opus 5.5` → `Claude Sonnet 5.5`)**: Cuts Anthropic execution cost by **70.0%** (\$2.51 vs \$8.35) with zero escalations and **2.3×** faster completion.
4. **`opus-flash` (`Claude Opus 5.5` → `Gemini 3.8 Flash`)**: Cross-provider hybrid pairing combining Opus 5.5 architectural planning with Flash execution speed, saving **76.8%** compared to running `opus-only`.

---

## Installation

```bash
git clone https://github.com/Bouboule0/Ai-Sup.git
cd Ai-Sup
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
cp .env.example .env
# Configure GEMINI_API_KEY and/or ANTHROPIC_API_KEY in .env
```

## CLI Usage

List all available AI pairings and their pricing:

```bash
orch pairings
```

Run the comparative multi-model benchmark:

```bash
orch benchmark
orch benchmark --output-md benchmark_results.md
```

Execute a goal with a specific preset pairing (`argon-flash`, `argon-pro`, `opus-sonnet`, `opus-flash`, etc.):

```bash
orch run --goal "Create a CSV to JSON CLI conversion script" --project ./examples/demo_project --pairing argon-flash --max-budget 2.0
```

Or specify any custom `--planner` and `--executor` combination directly:

```bash
orch run --goal "Refactor the utils module" --project ./my_project --planner claude-opus-5.5 --executor gemini-3.1-pro --max-budget 3.0
```

Generate a plan only:

```bash
orch plan --goal "Create a Python Fibonacci sequence module with tests" --planner gemini-4-argon
```

## Project Structure

```text
gemini-orchestrator/
├── pyproject.toml
├── README.md
├── .env.example
├── .gitignore
├── config/
│   ├── models.yaml
│   └── router.yaml
├── src/
│   └── orchestrator/
│       ├── __init__.py
│       ├── benchmark.py
│       ├── cli.py
│       ├── schemas.py
│       ├── planner/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── argon_planner.py
│       │   └── anthropic_planner.py
│       ├── executor/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── flash_executor.py
│       │   └── anthropic_executor.py
│       ├── router/
│       ├── tools/
│       ├── runtime/
│       └── utils/
├── examples/
│   └── demo_project/
│       └── README.md
└── tests/
    └── unit/
        ├── test_schemas.py
        ├── test_cost_tracker.py
        ├── test_router.py
        └── test_benchmark.py
```

## Cost Warning

Although `CostTracker` halts orchestration as soon as `--max-budget` is reached, a slight overrun on the final in-flight API call is possible depending on the size of the generated output. Always verify model pricing in `config/models.yaml` and set up billing alerts in Google Cloud / Anthropic Console.
