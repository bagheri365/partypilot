# PartyPilot v0.6d Three-Way LangChain Controlled Evaluation

- Benchmark: `PartyPilot v0.6d three-way LangChain controlled evaluation`
- Benchmark version: `1.0`
- Scenario count: `10`
- Run order blocks: `(('native_ollama', 'langchain_chatollama', 'langchain_agent'), ('langchain_chatollama', 'langchain_agent', 'native_ollama'), ('langchain_agent', 'native_ollama', 'langchain_chatollama'))`

## Provenance

- Experiment start Git SHA: `8c3cde626733eb200aaaff9a9e0a009ba56d40e1`
- Experiment start working tree dirty: `False`
- Experiment start git metadata error: `n/a`
- Artifact Git SHA: `8c3cde626733eb200aaaff9a9e0a009ba56d40e1`
- Artifact working tree dirty: `False`
- Artifact git metadata error: `n/a`
- Canonical start guard enforced: `True`
- Exploratory mode: `False`

## Reproducibility

- Git SHA: `8c3cde626733eb200aaaff9a9e0a009ba56d40e1`
- Working tree dirty: `False`
- Timestamp: `2026-08-13T04:26:41.218581+00:00`
- Python: `3.12.13`
- Model: `n/a`

## Variant Summaries

### native_ollama

- Runs: `3`
- Final decision accuracy: `mean=0.700, range=0.700..0.700`
- Evidence-grounded arbitration: `mean=0.625, range=0.625..0.625`
- Hard-constraint validity: `mean=0.600, range=0.600..0.600`
- Cross-domain compatibility: `mean=0.700, range=0.700..0.700`
- Global-optimum accuracy: `mean=1.000, range=1.000..1.000`
- Human-review calibration: `mean=1.000, range=1.000..1.000`
- Specialist success rate: `mean=0.187, range=0.160..0.200`
- Top-level specialist invocations: `150`
- Successful top-level specialist invocations: `28`
- Mean successful specialist latency (ms): `mean=22000.699, range=17702.783..26692.578`
- Median successful specialist latency (ms): `mean=22133.880, range=17702.783..26692.578`
- p95 successful specialist latency (ms): `25605.069`
- Mean scenario wall-clock latency (ms): `mean=48025.432, range=48014.766..48033.013`
- Maximum specialist latency (ms): `30080.507999999998`
- Total specialist invocations: `150`
- Successful specialist invocations: `28`
- Retry count: `0`
- Retry rate: `0.000`
- Terminal stability rate: `1.000`
- Specialist timeout outcomes: `122`
- Specialist timeout outcome rate: `0.813`
- Candidate specialist invocations: `240`
- Candidate provider timeout outcomes: `194`
- Candidate retry count: `0`
- Provider connection failure count: `0`
- Provider response failure count: `0`
- Structured-output validation failures: `0`
- Specialist-domain validation failures: `0`
- Provider attempts: `150`
- Provider attempt rate: `1.000`
- Tool calls: `n/a`
- No-tool specialist completions: `n/a`
- Scenarios with tool use: `none`
- Specialist domains with tool use: `none`
- Disposition: `BASELINE`

### langchain_chatollama

- Runs: `3`
- Final decision accuracy: `mean=0.700, range=0.700..0.700`
- Evidence-grounded arbitration: `mean=0.833, range=0.750..1.000`
- Hard-constraint validity: `mean=0.600, range=0.600..0.600`
- Cross-domain compatibility: `mean=0.700, range=0.700..0.700`
- Global-optimum accuracy: `mean=1.000, range=1.000..1.000`
- Human-review calibration: `mean=1.000, range=1.000..1.000`
- Specialist success rate: `mean=0.267, range=0.220..0.320`
- Top-level specialist invocations: `150`
- Successful top-level specialist invocations: `40`
- Mean successful specialist latency (ms): `mean=28892.430, range=17203.855..49663.402`
- Median successful specialist latency (ms): `mean=24851.474, range=17203.855..49663.402`
- p95 successful specialist latency (ms): `47427.932`
- Mean scenario wall-clock latency (ms): `mean=54122.374, range=52306.276..57480.713`
- Maximum specialist latency (ms): `49663.401999999995`
- Total specialist invocations: `150`
- Successful specialist invocations: `40`
- Retry count: `0`
- Retry rate: `0.000`
- Terminal stability rate: `1.000`
- Specialist timeout outcomes: `110`
- Specialist timeout outcome rate: `0.733`
- Candidate specialist invocations: `240`
- Candidate provider timeout outcomes: `180`
- Candidate retry count: `0`
- Provider connection failure count: `0`
- Provider response failure count: `0`
- Structured-output validation failures: `0`
- Specialist-domain validation failures: `0`
- Provider attempts: `150`
- Provider attempt rate: `1.000`
- Tool calls: `n/a`
- No-tool specialist completions: `n/a`
- Scenarios with tool use: `none`
- Specialist domains with tool use: `none`
- Disposition: `RETAIN`

### langchain_agent

- Runs: `3`
- Final decision accuracy: `mean=0.700, range=0.700..0.700`
- Evidence-grounded arbitration: `mean=0.625, range=0.625..0.625`
- Hard-constraint validity: `mean=0.600, range=0.600..0.600`
- Cross-domain compatibility: `mean=0.700, range=0.700..0.700`
- Global-optimum accuracy: `mean=1.000, range=1.000..1.000`
- Human-review calibration: `mean=1.000, range=1.000..1.000`
- Specialist success rate: `mean=0.233, range=0.200..0.280`
- Top-level specialist invocations: `150`
- Successful top-level specialist invocations: `35`
- Mean successful specialist latency (ms): `mean=27429.927, range=19671.909..44075.489`
- Median successful specialist latency (ms): `mean=25628.032, range=19671.909..44075.489`
- p95 successful specialist latency (ms): `42714.094000000005`
- Mean scenario wall-clock latency (ms): `mean=50575.062, range=49183.029..53031.143`
- Maximum specialist latency (ms): `44075.488999999994`
- Total specialist invocations: `150`
- Successful specialist invocations: `35`
- Retry count: `0`
- Retry rate: `0.000`
- Terminal stability rate: `1.000`
- Specialist timeout outcomes: `115`
- Specialist timeout outcome rate: `0.767`
- Candidate specialist invocations: `240`
- Candidate provider timeout outcomes: `187`
- Candidate retry count: `0`
- Provider connection failure count: `0`
- Provider response failure count: `0`
- Structured-output validation failures: `0`
- Specialist-domain validation failures: `0`
- Provider attempts: `150`
- Provider attempt rate: `1.000`
- Tool calls: `0`
- No-tool specialist completions: `35`
- Scenarios with tool use: `none`
- Specialist domains with tool use: `none`
- Disposition: `REJECT_AS_DEFAULT`

## Runs

### 20260813T042641Z-1-1

- Variant: `native_ollama`
- Repetition: `1`
- Order block: `1`
- Order position: `1`
- Run report path: `run.json` / `run.md`

### 20260813T042641Z-1-2

- Variant: `langchain_chatollama`
- Repetition: `1`
- Order block: `1`
- Order position: `2`
- Run report path: `run.json` / `run.md`

### 20260813T042641Z-1-3

- Variant: `langchain_agent`
- Repetition: `1`
- Order block: `1`
- Order position: `3`
- Run report path: `run.json` / `run.md`

### 20260813T042641Z-2-1

- Variant: `langchain_chatollama`
- Repetition: `2`
- Order block: `2`
- Order position: `1`
- Run report path: `run.json` / `run.md`

### 20260813T042641Z-2-2

- Variant: `langchain_agent`
- Repetition: `2`
- Order block: `2`
- Order position: `2`
- Run report path: `run.json` / `run.md`

### 20260813T042641Z-2-3

- Variant: `native_ollama`
- Repetition: `2`
- Order block: `2`
- Order position: `3`
- Run report path: `run.json` / `run.md`

### 20260813T042641Z-3-1

- Variant: `langchain_agent`
- Repetition: `3`
- Order block: `3`
- Order position: `1`
- Run report path: `run.json` / `run.md`

### 20260813T042641Z-3-2

- Variant: `native_ollama`
- Repetition: `3`
- Order block: `3`
- Order position: `2`
- Run report path: `run.json` / `run.md`

### 20260813T042641Z-3-3

- Variant: `langchain_chatollama`
- Repetition: `3`
- Order block: `3`
- Order position: `3`
- Run report path: `run.json` / `run.md`
