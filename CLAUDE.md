# CLAUDE.md

Contributor/agent-facing guide to this repository. The `README.md` is the user-facing
documentation (step-by-step usage, IAM, dashboards) — this file complements it with
architecture, conventions, and the non-obvious facts you need before changing anything.

## Project overview

This is an AWS sample (`sample-ray-on-amazon-sagemaker-training-jobs`) showing how to run
**Ray** distributed workloads inside **Amazon SageMaker training jobs**. The centerpiece is
`scripts/launcher.py`: a universal entrypoint that bootstraps a Ray cluster on the SageMaker
instances allocated to a job and then runs the user's training/inference script with Ray
already initialized. Everything else in the repo (notebooks, example scripts, Grafana
dashboard) demonstrates how to use that launcher.

## Repository structure

```
scripts/
  launcher.py              # CANONICAL launcher — the single source of truth
  requirements.txt         # ray[data,train,tune,serve]==2.56.1, sagemaker==3.20.0
examples/<pattern>/<framework>/
  notebook.ipynb           # copies launcher.py in, then launches a ModelTrainer job
  scripts/                 # entry script + model + requirements.txt
grafana-dashboards/
  ray_sagemaker_training_dashboard.json
images/  README.md  CONTRIBUTING.md  LICENSE  CODE_OF_CONDUCT.md
```

Four Ray patterns, each with a homogeneous and a heterogeneous variant:

| Pattern            | Entry script   | What it shows                                    |
| ------------------ | -------------- | ------------------------------------------------ |
| `ray-remote`       | `train.py`     | `@ray.remote` task parallelism (sentiment clf)   |
| `ray-data`         | `inference.py` | Batch inference with `ray.data` (ResNet152)      |
| `ray-tune`         | `tune.py`      | HPO with ASHA on CIFAR-10                        |
| `ray-torchtrainer` | `train_ray.py` | LLM LoRA/QLoRA fine-tuning (HuggingFace + Train) |

Variant dirs are named `pytorch` / `pytorch-heterogeneous` (or `huggingface` /
`huggingface-heterogeneous` for `ray-torchtrainer`).

## The launcher (`scripts/launcher.py`)

**Single source of truth.** The README states users should not modify the launcher — it is a
universal entrypoint. Any change to launcher behavior belongs in `scripts/launcher.py`.

The same `launcher.py` runs on **every** instance in the job. It does not know in advance
whether it is the head or a worker — it discovers its role at runtime from the SageMaker
environment (`env.current_host` vs the computed `head_host`) and then follows the matching code
path. This is why a single file can drive single-node, homogeneous multi-node, and
heterogeneous clusters.

### File organization (top to bottom)

Line ranges are indicative and drift with every change — grep for the symbol rather than
trusting the number.

| Lines     | Section                | Contents                                                               |
| --------- | ---------------------- | ---------------------------------------------------------------------- |
| 1–29      | Imports                | stdlib, `boto3`, `sagemaker_training`, `ray`, `requests`, `yaml`       |
| 32–59     | Logging                | `get_logger()` — single configured, non-propagating logger             |
| 62–92     | Constants & globals    | exit codes, ports, timeouts, `FAILURE_REASON_PATH`, mutable globals    |
| 95–123    | Signal handling        | `signal_handler()` + SIGTERM/SIGINT registration                       |
| 126–230   | EFA discovery          | `get_efa_supported_gpu_instances()` + fallback/RDMA instance lists     |
| 233–482   | Arg/env parsing        | `_parse_args()` — CLI args, env-var fallbacks, entrypoint splitting    |
| 485–849   | Prometheus helpers     | binary copy/extract, remote-write URL building, config/label injection |
| 852–1167  | Grafana helpers        | archive extract, config provisioning, launch/readiness/shutdown        |
| 1170–1204 | EFA device detection   | `_is_efa_device_present()` — real device check, not just type          |
| 1207–1344 | Runtime env builder    | `_create_runtime_environment()` — env vars passed to Ray workers       |
| 1347–1506 | Entry-script execution | `_execute_entry_script()` + `.py` (importlib) / `.sh` (subprocess)     |
| 1509–1859 | Process/util helpers   | host→IP, log readers, subprocess runners, command allowlist, untar     |
| 1862–2126 | Node setup             | `_setup_head_node()`, `_setup_worker_node()`                           |
| 2129–2227 | Cluster topology       | `_is_ray_alive()`, `_get_cluster_configuration()` + homo/hetero        |
| 2230–2551 | Orchestration          | single-node / multi-node setup + homo/hetero environment entrypoints   |
| 2554–2645 | Main                   | `_write_failure_reason_file()`, `main()`, `__main__` guard             |

### Control flow

`main()` → `_parse_args()` → `sagemaker_training.environment.Environment()` → optional
Prometheus/Grafana binary extraction → branch on `env.is_hetero`:

- `_setup_ray_environment_homogeneous_cluster` / `_setup_ray_environment_heterogeneous_cluster`
- both call `_create_runtime_environment`, then `_get_cluster_configuration` → `(all_hosts, head_host, compute_host_count)`
- **1 instance (`len(all_hosts) == 1`)** → `_setup_single_node_ray`; else `_setup_multi_node_ray`
- multi-node routes by `env.current_host == head_host` → `_setup_head_node` or `_setup_worker_node`

```
main
 ├─ _parse_args                      args + env-var fallbacks; sets source_dir/entry_script
 ├─ Environment()                    SageMaker cluster metadata (hosts, instance type, is_hetero)
 ├─ _copy_prometheus_binary          (only if --prometheus-path) extract offline binary
 ├─ _copy_grafana_binary             (only if --grafana-path + dashboard) extract Grafana
 └─ _setup_ray_environment_*_cluster
     ├─ _create_runtime_environment  build env-var dict for Ray (NCCL/EFA/RDMA/Prometheus/Grafana)
     ├─ _get_cluster_configuration   → (all_hosts, head_host, compute_host_count)
     ├─ len(all_hosts)==1 → _setup_single_node_ray ─┐
     └─ else              → _setup_multi_node_ray   │
                    ├─ head → _setup_head_node ──┤ both: ray start → ray.init →
                    └─ else → _setup_worker_node │   [Prometheus/Grafana] → _run_script → shutdown
```

### Phase-by-phase behavior

1. **Parse & normalize inputs** (`_parse_args`). Reads CLI flags, then fills any unset
   value from the matching env var. Splits `-e/--entrypoint` (e.g. `training/train.py`) into
   `source_dir` + `entry_script` and exports them as env vars for later phases. Unknown args are
   logged and ignored (so user training args pass through harmlessly).
2. **Read SageMaker environment** (`main`). `Environment()` exposes `hosts`,
   `current_host`, `instance_groups_dict`, `is_hetero`, `num_cpus`, `num_gpus`,
   `network_interface_name`, `current_instance_type`. SageMaker also sets `TRAINING_JOB_NAME`
   in the container, which the launcher uses as a Prometheus label.
3. **Build the Ray runtime env** (`_create_runtime_environment`). Copies the full process
   environment, prepends `source_dir` to `PYTHONPATH`, sets NCCL vars, enables EFA/RDMA only
   when the instance type is capable **and** an EFA device is actually present, and wires
   Prometheus/Grafana env vars. This dict is passed to
   `ray.init(runtime_env={"env_vars": ...})` so every Ray worker inherits it.
4. **Resolve topology** (`_get_cluster_configuration`) → `(all_hosts, head_host,
compute_host_count)`. Homogeneous: head = `hosts[0]`. Heterogeneous: head = first host of
   `--head-instance-group`. `compute_host_count` excludes a coordinator-only head
   (`head_num_cpus==0 and head_num_gpus==0`) and is **observability only** — the single vs
   multi-node decision uses `len(all_hosts)`. See the topology section below.
5. **Start Ray for this node's role** (head/worker/single). See below.
6. **Run the user script** (`_run_script` → `_execute_entry_script`) — only on the head /
   single node, after the cluster is assembled.
7. **Shut down** in a `finally` block: optional `--wait-shutdown` delay, `ray metrics
shutdown-prometheus`, `ray.shutdown()`, `ray stop`. On the worker, instead loop on `ray
status` until the head disappears, then `ray stop`.

### Head node, in detail (`_setup_head_node`)

1. Compute the head's resources: `num_cpus`/`num_gpus` default to `env.num_cpus`/`env.num_gpus`
   but are overridden by `--head-num-cpus`/`--head-num-gpus` when provided. These can be
   `0` for coordinator-only heterogeneous heads.
2. Build the start command:
   `ray start --head --num-cpus=<n> --num-gpus=<n> --port=6379`, always appending
   `--metrics-export-port=8080`, plus `--dashboard-host=0.0.0.0 --dashboard-port=8265` only when
   `--include-dashboard` is on. Values are `shlex.quote`d.
3. Run it via `_run_subprocess_command_with_env` (the runtime-env dict is merged into the Ray
   process environment), then `ray.init(address="auto", include_dashboard=..., runtime_env={"env_vars": runtime_env})`.
4. **Prometheus** (only if `launch_prometheus` — the Dashboard is not required): if a remote host
   was detected, inject `remote_write` into the config _before_ launch; also inject the
   `instance_type` / `sagemaker_training_job_name` relabels. Then start either the
   custom binary (`_build_prometheus_command`) or `ray metrics launch-prometheus`, async, with
   stdout/stderr redirected to `/tmp/prometheus_*.log`. Poll `GET <host>/-/healthy` every 2s up
   to `PROMETHEUS_WAIT_SECONDS=300`, bailing early if the process exits (which usually means no
   internet to download the binary). Failure here is logged, not fatal.
5. **Grafana** (only if `grafana_path` — the Dashboard is not required): `_launch_grafana` spawns the
   process and returns immediately — readiness is confirmed by `_wait_for_grafana_ready`
   _after_ the worker-wait below, so Grafana startup overlaps work the head must do anyway.
6. **Wait for workers**: loop reading `ray.available_resources()`, counting keys that
   start with `node:` (excluding the synthetic `node:__internal_head__`), until
   `connected_nodes == cluster_size` (`cluster_size = len(all_hosts)`) or
   `RAY_CONNECTION_TIMEOUT=300s` elapses (after which it proceeds with whatever connected,
   logging a warning). A coordinator-only head still registers as a node here.
7. `_run_script(runtime_env)` executes the user entry script.
8. **`finally` teardown**: if no failure, honor `--wait-shutdown`; then `_shutdown_grafana()`,
   `ray metrics shutdown-prometheus`, `_shutdown_ray_safely()` (`ray.shutdown()`), and
   `ray stop`. Returns the `ray stop` return code. Note the `return` inside `finally`
   swallows in-flight exceptions — correctness relies on the `has_failure` global.

### Worker node, in detail (`_setup_worker_node`)

1. Resolve the head hostname to an IP with `_get_ip_from_host`: `socket.gethostbyname`
   retried up to 200 times with a 5s sleep between attempts (~1000s max; SageMaker DNS can lag
   at startup).
2. `ray start --address=<ip>:6379` with the runtime env merged in.
3. Stay alive: loop `_is_ray_alive()` (which shells out to `ray status`) every
   `RAY_WORKER_POLL_INTERVAL=10s`, logging roughly once a minute, until the head disappears.
4. Then `ray stop` and return.

Workers never run the user script, Prometheus or Grafana — they only join the cluster and keep
the process alive so SageMaker doesn't tear the instance down early.

### Single-node, in detail (`_setup_single_node_ray`)

Same shape as the head path but without the worker-wait loop: `ray start --head --port=6379`
(+ dashboard), `ray.init`, optional Prometheus/Grafana, `_run_script`, then the same `finally`
teardown. Chosen when `len(all_hosts) == 1`. The single node must contribute compute, so an
explicit `head_num_cpus=0` / `head_num_gpus=0` is deliberately **ignored** here (with a
warning) rather than honored, which would leave nothing able to run a task.

### Cluster topology resolution (`_get_cluster_configuration`)

Returns `(all_hosts, head_host, compute_host_count)`.

- **Homogeneous** (`_get_homogeneous_cluster_config`): `head_host = hosts[0]`,
  `compute_host_count = len(hosts)`.
- **Heterogeneous** (`_get_heterogeneous_cluster_config`): iterate
  `env.instance_groups_dict`, accumulate all hosts; the head is the first host of the group
  named by `--head-instance-group`. If that head is coordinator-only
  (`head_num_cpus==0 and head_num_gpus==0`) it is excluded from `compute_host_count` (other
  hosts in the head group still count, they join as normal workers). Raises if the named group
  isn't found.

**`compute_host_count` must never drive the topology decision.** It is a compute-capacity
number reported for observability; the single vs multi-node branch uses `len(all_hosts)`, and
the head-wait loop uses `len(all_hosts)` too. Using the compute count made a coordinator head
plus a single worker look single-node on _both_ instances, so each started its own Ray cluster
and ran the entry script twice (same for a 2-instance coordinator head group with no worker
group). The subtraction has no other consumer.

### User entry script invocation (`_execute_entry_script`)

Driven by env vars `source_dir` / `entry_script` (set from `-e/--entrypoint` in `_parse_args`).
Runs only on the head / single node, _after_ the cluster is assembled, so Ray is already
initialized when user code runs.

- Resolves the absolute script path under the SageMaker code dir (`/opt/ml/input/data/code`),
  logging directory contents for debugging, and raises `FileNotFoundError` if missing.
- `.py` → `_execute_python_script`: `chdir` into the source dir, prepend it to
  `sys.path`, then load via `importlib.util.spec_from_file_location("__main__", path)` and
  `exec_module`. The module **must** guard real work behind `if __name__ == "__main__":`.
- `.sh` → `_execute_bash_script`: `subprocess.run(["bash", path], env=runtime_env,
check=True, capture_output=True)`; stdout/stderr are logged.
- Any other extension → `ValueError`. On exception, sets `has_failure = True`, logs the
  traceback, and re-raises; a `finally` restores the original working directory.

### Runtime environment construction (`_create_runtime_environment`)

Signature is `(args, env)`. Builds the env-var dict handed to
`ray.init(runtime_env={"env_vars": ...})`, so it propagates to every Ray worker process across
the cluster:

- Starts from a full copy of the launcher's `os.environ`.
- Prepends the absolute `source_dir` to `PYTHONPATH` so user modules import on all nodes.
- Sets `NCCL_SOCKET_IFNAME = env.network_interface_name` and `NCCL_PROTO = simple`.
- **EFA is device-gated, not type-inferred.** `FI_PROVIDER=efa` is set only when
  `current_instance_type ∈ SM_EFA_NCCL_INSTANCES` **and** `_is_efa_device_present()` confirms a
  real device (`fi_info -p efa`, falling back to `/sys/class/infiniband`). An EFA-capable type
  can have no EFA attached for a given job (e.g. a coordinator-only head), and forcing
  `FI_PROVIDER=efa` there points libfabric at a device that does not exist.
- `SM_EFA_RDMA_INSTANCES` (p4d/p4de/trn1) → `FI_EFA_USE_DEVICE_RDMA=1`, `RDMAV_FORK_SAFE=1`,
  again only when a device is actually present.
- An explicit `FI_PROVIDER` / `FI_EFA_USE_DEVICE_RDMA` / `RDMAV_FORK_SAFE` from the ModelTrainer
  `environment` dict always wins over autodetection.
- Prometheus/Grafana wiring: when `launch_prometheus`, sets `RAY_PROMETHEUS_HOST` to
  `http://127.0.0.1:9090` for the Dashboard, and if the user-supplied `RAY_PROMETHEUS_HOST` is
  remote, stores it as `RAY_REMOTE_WRITE_PROMETHEUS_HOST` (plus passes through
  `RAY_PROMETHEUS_USERNAME`/`PASSWORD`). Also forwards `RAY_PROMETHEUS_NAME`, `RAY_GRAFANA_HOST`,
  and `RAY_GRAFANA_IFRAME_HOST` (defaulting the iframe host to `RAY_GRAFANA_HOST`).
  **Note:** when `launch_prometheus` is false, a user-supplied `RAY_PROMETHEUS_HOST` is passed
  straight through, so the Dashboard queries that endpoint _from inside the job_ — which is why
  the execution role needs AMP read actions, not just `aps:RemoteWrite`.

The EFA-capable instance list is fetched live from EC2 `describe_instance_types` at _import
time_ (`get_efa_supported_gpu_instances`), filtered to GPU instances, with a static
fallback (`SM_EFA_NCCL_INSTANCES_FALLBACK`) if credentials/API fail. This needs
`ec2:DescribeInstanceTypes`; without it the call fails silently into the static list.

### Prometheus / observability

Local Prometheus runs on `127.0.0.1:9090` and the Ray Dashboard points at it. Remote write is
controlled by `RAY_PROMETHEUS_HOST`. The flow on the head/single node:

1. `_extract_amp_region` regex-matches `aps-workspaces.<region>.amazonaws.com`.
2. `_build_remote_write_url` appends `/api/v1/remote_write` for AMP (region matched) or
   `/api/v1/write` for self-hosted, preserving any existing path on the host.
3. `_inject_remote_write_config` loads the prometheus.yml, appends a `remote_write` entry
   with `queue_config` (max_samples_per_send 1000, max_shards 200, capacity 2500), adds a
   `sigv4: {region}` block for AMP or a `basic_auth` block for self-hosted, and writes it back.
   Injection happens **before** Prometheus launches.
4. `_inject_sagemaker_relabels` adds two **scrape-time** `relabel_configs` to Ray's scrape job:
   `instance_type` (per node, matched on the target's `__address__`, built by
   `_build_ip_instance_type_map`) and `sagemaker_training_job_name` (constant, from the
   `TRAINING_JOB_NAME` env var SageMaker sets). Both back dashboard variables
   (`InstanceType`, `TrainingJobName`).

   **Why relabels and not `external_labels`:** `external_labels` apply only to `remote_write`
   samples, so the label would be missing on the local Prometheus that backs the Dashboard —
   and therefore missing for anyone not using AMP. Relabels run before ingestion _and_ before
   remote_write, so they land in both. Do not "simplify" this back into a `global:
external_labels` block.

**Embedded Grafana** (`_launch_grafana`). Opt-in via `--grafana-path` / `grafana_path`, for clusters
that can reach neither an external Prometheus nor a Grafana instance. `_copy_grafana_binary`
extracts the archive into `/opt/ml/code` — the folder name is read from the archive
(`_get_archive_root_folder`) because it does not match the archive name
(`grafana-12.0.1.linux-amd64.tar.gz` → `grafana-v12.0.1`). Grafana is then started from the config
the Ray Dashboard generates in `RAY_GRAFANA_CONFIG_DIR`
(`/tmp/ray/session_latest/metrics/grafana`), which already carries `allow_embedding`, anonymous
auth, the Prometheus datasource and the Ray dashboards with the UIDs the Metrics tab embeds;
`_write_grafana_fallback_config` covers the case where Ray never wrote it.
`_build_grafana_command` prefers `bin/grafana server` (Grafana 10+) over the deprecated
`bin/grafana-server`. `_create_runtime_environment` defaults `RAY_GRAFANA_HOST` to loopback and
`RAY_GRAFANA_IFRAME_HOST` to `localhost` (the browser reaches it through SSM port forwarding),
never overriding user-provided values. The whole path is best effort: every failure is logged, none
fails the job. Launch and readiness are split (`_launch_grafana` spawns,
`_wait_for_grafana_ready` polls `/api/health`) so the wait overlaps the worker-wait instead of
delaying job start; `_shutdown_grafana` terminates it in the `finally` teardown.

`_provision_repo_dashboard` additionally installs this repo's
`ray_sagemaker_training_dashboard.json` into a `SageMaker` folder, on top of whichever config is
in place. Only `source_dir` is uploaded to a training job, so the JSON is found only when the
user shipped it with their code — `_find_repo_dashboard` checks `grafana_dashboard_path` first,
then `source_dir` / `/opt/ml/code` / `/opt/ml/input/data/code` and a `grafana-dashboards/`
subfolder of each. Absence is logged as info, not an error.

**The Dashboard, Prometheus and Grafana are three independent switches.** `include_dashboard`
controls only the Ray Dashboard UI (`--dashboard-host`/`--dashboard-port` and
`ray.init(include_dashboard=...)`). Prometheus is gated on `launch_prometheus` alone and Grafana on
`grafana_path` alone — including extraction and teardown. Do not re-couple them.

`--metrics-export-port=8080` is therefore appended to `ray start` **unconditionally**: the exporter
lives in the per-node agent, not the Dashboard UI, so metrics do not need the UI. What the Dashboard
_does_ own is the scrape config and the service-discovery file Ray's config points at — so when
`include_dashboard` is false, `_write_prometheus_static_config` writes a fresh prometheus.yml whose
`static_configs` targets are `<ip>:8080` for every node, reusing the IPs
`_build_ip_instance_type_map` already resolves. Its scrape job is named `ray` precisely so
`_inject_remote_write_config` and `_inject_sagemaker_relabels` keep working on it untouched. This
makes "no UI, remote_write to AMP" a supported setup.

Grafana renders whatever `RAY_PROMETHEUS_HOST` resolves to, which is why `launch_prometheus=false`
with no host yields empty panels — `_parse_args` warns about that, and about an AMP host combined
with basic-auth credentials. Without the Dashboard, `_launch_grafana` skips
`_wait_for_ray_grafana_config` entirely (that file will never appear, so waiting
`GRAFANA_CONFIG_WAIT_SECONDS` for it is pure delay) and goes straight to the fallback config; the
only thing lost is the embedded Metrics tab, so Grafana is reached by port-forwarding its port.
Setting a non-loopback `RAY_GRAFANA_HOST` clears `grafana_path`, since the embedded server would
just be an unqueried process.

AMP needs SigV4, and **two things are required** — this is the trap.
`_enable_amp_sigv4_on_datasources` adds `jsonData.sigV4Auth` + region to every provisioned
datasource whose URL matches an AMP workspace (Ray's own generated datasource included, since Ray
has no notion of AMP), **and** `_launch_grafana` exports `GF_AUTH_SIGV4_AUTH_ENABLED=true` plus
`AWS_SDK_LOAD_CONFIG=true`, because Grafana ships with SigV4 support disabled and silently ignores
the datasource flag without them. Setting only the datasource flag looks correct and still returns
403 on every panel. Signing uses the container's credentials, i.e. the execution role, which
therefore needs the AMP read actions — and the job must be able to reach AMP (VPC endpoint).

Config-path selection (`_get_prometheus_config_path`): `ray metrics launch-prometheus`
uses the Ray package template path; a custom binary uses the session config
`/tmp/ray/session_latest/metrics/prometheus/prometheus.yml`. For offline clusters, pass a
pre-downloaded binary via `--prometheus-path` — `_copy_prometheus_binary` copies the
tar.gz into `/opt/ml/code`, verifies size, and extracts it with the traversal-safe
`_safe_extract_all`. Both archives are unpacked by `_extract_observability_binaries`, called from
the head and single-node paths **only** — workers never launch either process, so extracting a few
hundred MB there was pure waste. A bad Prometheus archive is fatal (it was requested explicitly);
Grafana stays best effort.

`_inject_remote_write_config` emits **at most one** auth mechanism: Prometheus rejects a
`remote_write` entry carrying both `sigv4` and `basic_auth`, and would exit at startup leaving the
job with no metrics, so an AMP region wins and basic auth is dropped with a warning.

### Subprocess execution helpers

All shelling-out is centralized and hardened:

- `_validate_command`: allowlist — only `ray`, `bash`, `./prometheus-*`,
  `/opt/ml/code/prometheus-*`, or a Grafana binary (`_is_grafana_executable`) may run; anything
  else raises `ValueError`. `_is_grafana_executable` requires the basename to be
  `grafana`/`grafana-server` **and** the parent to be exactly the `bin/` of the folder
  `_copy_grafana_binary` extracted — matching the real folder rather than a `grafana-` name
  prefix, so a repackaged archive is not silently refused.
- `_run_subprocess_command` / `_run_subprocess_command_with_env`: `shlex.split`
  the command, validate it, run with `shell=False`, capture output, optionally `check`.
- `_run_subprocess_command_async`: `Popen` for long-running processes (Ray start,
  Prometheus, Grafana), with optional `/tmp/`-restricted stdout/stderr files and an optional
  post-start wait. It copies `os.environ` and merges `env_vars` on top, so callers only pass the
  variables they want to override.

### Failure & shutdown semantics

- SIGTERM/SIGINT → `signal_handler` shuts Ray down and exits with `DEFAULT_FAILURE_CODE`
  if `has_failure` else `SUCCESS_EXIT_CODE`. It does not stop Grafana directly, but
  `sys.exit()` raises `SystemExit` through the setup functions, so their `finally` blocks (and
  `_shutdown_grafana`) still run.
- Errors set the module global `has_failure = True` and propagate. `main` then writes
  `/opt/ml/output/failure` via `_write_failure_reason_file` (only if the file doesn't
  already exist) and raises `RuntimeError`, ensuring **SageMaker marks the job FAILED** rather
  than silently succeeding.
- Module-level mutable globals coordinate this across functions: `ray_initialized`,
  `has_failure`, `prometheus_folder_name`, `grafana_folder_name`, `grafana_process`.

### Key constants

| Constant                      | Value                                     | Meaning                                |
| ----------------------------- | ----------------------------------------- | -------------------------------------- |
| `DEFAULT_RAY_PORT`            | 6379                                      | Ray GCS port                           |
| `RAY_DASHBOARD_PORT`          | 8265                                      | Ray Dashboard UI (gated on the flag)   |
| `RAY_METRICS_EXPORT_PORT`     | 8080                                      | per-node metrics exporter (always on)  |
| `RAY_CONNECTION_TIMEOUT`      | 300s                                      | head waits for workers                 |
| `RAY_WORKER_POLL_INTERVAL`    | 10s                                       | worker liveness poll                   |
| `PROMETHEUS_WAIT_SECONDS`     | 300                                       | Prometheus readiness wait              |
| `DEFAULT_GRAFANA_PORT`        | 3000                                      | embedded Grafana HTTP port             |
| `GRAFANA_WAIT_SECONDS`        | 120                                       | Grafana readiness wait                 |
| `GRAFANA_CONFIG_WAIT_SECONDS` | 60                                        | wait for Ray's Grafana config          |
| `RAY_GRAFANA_CONFIG_DIR`      | `/tmp/ray/session_latest/metrics/grafana` | Ray-generated Grafana config           |
| `RESERVED_PORTS`              | 6379/8080/8265/9090                       | ports the embedded Grafana may not use |
| `REPO_DASHBOARD_FILENAME`     | `ray_sagemaker_training_dashboard.json`   | dashboard provisioned when found       |

## Conventions & gotchas

- **Edit only `scripts/launcher.py` for launcher changes.** It is the canonical copy.
- **Any `examples/*/scripts/launcher.py` you see locally is an untracked build artifact.**
  `.gitignore` ignores `examples/**/launcher.py`, so `git ls-files` returns none of them — they
  are left behind by notebook runs, which copy `scripts/launcher.py` into the example `scripts/`
  dir. A fresh clone has none. Never hand-edit them, and don't treat a stale local copy as
  evidence of drift.
- **Homogeneous and heterogeneous variants share byte-identical entry scripts.** The whole
  hetero adaptation lives in the notebook's `environment` dict (`head_instance_group`,
  `head_num_cpus: "0"`, `head_num_gpus: "0"`). If you change an entry script, copy it to its
  sibling variant so they stay identical.
- **Most CLI args have an env-var fallback** with the same name (`wait_shutdown`,
  `head_instance_group`, `head_num_cpus`, `head_num_gpus`, `launch_prometheus`,
  `prometheus_path`, `grafana_path`, `grafana_port`) via the ModelTrainer `environment` dict.
  Two exceptions: `--include-dashboard` has **no** fallback, and `-e/--entrypoint` has none
  (but is equivalent to setting `source_dir` + `entry_script` directly).
- **Env-var fallbacks must not override an explicit CLI value.** The pattern is
  `if args.X is None: <read env>`, with the default applied afterwards — never
  `if args.X == DEFAULT`, which cannot distinguish "unset" from "explicitly set to the
  default". `launch_prometheus` is the deliberate exception: its CLI default is `True`, so
  argparse cannot distinguish "unset" from "explicitly true" and it reads its env var
  unconditionally — making `launch_prometheus` an **override** rather than a fallback, which is
  useful when `command` is templated and `environment` is the per-job knob. Only a
  self-contradictory config (both set, disagreeing) diverges, and the choice is logged. Don't
  copy this for new args that can use `default=None`.
- **Driver vs worker device checks.** The head is frequently a CPU-only coordinator, so never
  decide GPU placement on the driver: `torch.cuda.is_available()` there reports the _head's_
  hardware. Size trial/worker resources from `ray.cluster_resources()`, and resolve `device`
  inside the remote function.
- **Ray Tune/Train `storage_path` must be shared.** `/opt/ml/output/data` is node-local on
  SageMaker, so multi-node runs need an S3 URI (`--storage_path` for `tune.py`,
  `RAY_STORAGE_PATH` for `train_ray.py`); both warn when they fall back on a >1-node cluster.
- **Prefer short env-var names** (e.g. `RAY_PROMETHEUS_USERNAME`, not
  `RAY_PROMETHEUS_REMOTE_WRITE_USERNAME`).
- **No custom Dockerfile** — runs on the stock SageMaker PyTorch container; deps come from
  `requirements.txt`.
- **Ray Autoscaler is not supported** — SageMaker cluster size is fixed at job creation.

## Common tasks

- **Change launcher behavior** → edit `scripts/launcher.py` only; ignore the example copies.
- **Add an example** → mirror an existing pattern dir: a `notebook.ipynb` plus
  `scripts/{entry script, model.py, requirements.txt}`. The notebook should copy
  `scripts/launcher.py` in and launch via the PySDK v3 `ModelTrainer` with
  `command="python launcher.py -e <script> ..."`.
- **Run an example** → open its `notebook.ipynb`; it handles data prep, launcher copy, and the
  `ModelTrainer.train()` call. See README "Example Usage" for the full snippet.

## Parameters & dependencies

Full parameter and environment-variable tables live in README → "Required Parameters and
Environment Variables" (kept there to avoid duplication/drift). Core dependency pins are in
`scripts/requirements.txt` (`ray[data,train,tune,serve]==2.56.1`, `sagemaker==3.20.0`);
per-example extras live in each example's `scripts/requirements.txt`.
