import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Cpu,
  ExternalLink,
  Gauge,
  Play,
  RadioTower,
  RefreshCcw,
  Server,
  ShieldCheck,
  SlidersHorizontal,
  StopCircle,
} from "lucide-react";
import { Fragment, useEffect, useMemo, useState } from "react";

const DEFAULT_ENGINES = ["vllm", "sglang", "ollama"];
const STATUS_CLASS = {
  Validated: "is-valid",
  "Needs SSH": "is-muted",
  "Unproven on this GPU": "is-warn",
  "Legacy profile": "is-legacy",
  Failed: "is-fail",
};

async function readJson(response, fallbackDetail) {
  try {
    return await response.json();
  } catch {
    return { detail: fallbackDetail };
  }
}

function parseEventPayload(data) {
  try {
    return normalizeEvent(JSON.parse(data));
  } catch {
    return null;
  }
}

function parseJobPayload(data) {
  try {
    return normalizeJob(JSON.parse(data));
  } catch {
    return null;
  }
}

function normalizeJob(payload) {
  const job = payload && typeof payload === "object" ? payload : {};
  return {
    ...job,
    job_id: toDisplayText(job.job_id || "unknown job"),
    requested_engines: Array.isArray(job.requested_engines) ? job.requested_engines : [],
    status: toDisplayText(job.status || "failed"),
    events: eventsOf(job),
    results: Array.isArray(job.results) ? job.results.filter(Boolean) : [],
    error: job.error ? toDisplayText(job.error) : null,
  };
}

function normalizeEvent(payload) {
  const event = payload && typeof payload === "object" ? payload : {};
  const index = Number(event.index);
  return {
    ...event,
    index: Number.isFinite(index) ? index : Date.now(),
    type: toDisplayText(event.type || "event"),
    message: toDisplayText(event.message || ""),
    details: event.details && typeof event.details === "object" ? event.details : {},
  };
}

function eventsOf(job) {
  return Array.isArray(job?.events) ? job.events.map(normalizeEvent) : [];
}

function failCurrentJob(current, message) {
  const currentEvents = eventsOf(current);
  const displayMessage = toDisplayText(message || "Run failed.");
  return {
    ...(current || {}),
    status: "failed",
    error: displayMessage,
    events: [
      ...currentEvents,
      { index: currentEvents.length, type: "job_failed", message: displayMessage },
    ],
    results: Array.isArray(current?.results) ? current.results : [],
  };
}

function toDisplayText(value) {
  if (typeof value === "string") return value;
  if (value == null) return "";
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function finiteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function App() {
  const [catalog, setCatalog] = useState({ engines: [] });
  const [sshTarget, setSshTarget] = useState("");
  const [preflight, setPreflight] = useState(null);
  const [selectedEngines, setSelectedEngines] = useState(DEFAULT_ENGINES);
  const [parameters, setParameters] = useState({
    modelProfile: "qwen35-2b",
    prompt: "Explain local model serving in one concise sentence.",
    batchPrompts: "",
    totalRequests: 1,
    concurrentUsers: 1,
    requestRateRps: 0.1,
    warmupRequests: 1,
    kvCacheMode: "engine_default",
    maxTokens: 32,
    temperature: 0,
    topP: 1,
    timeoutSeconds: 180,
  });
  const [job, setJob] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetch("/api/engines")
      .then((response) => response.json())
      .then(setCatalog)
      .catch(() => setCatalog({ engines: [] }));
  }, []);

  const recommendations = useMemo(() => {
    const items = preflight?.recommendations || [];
    return Object.fromEntries(items.map((item) => [item.engine, item]));
  }, [preflight]);

  async function runPreflight() {
    setBusy(true);
    try {
      const response = await fetch("/api/preflight", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sshTarget: sshTarget || null }),
      });
      const payload = await readJson(response, "GPU check could not complete.");
      if (!response.ok) {
        setPreflight({
          status: "Failed",
          gpu: null,
          recommendations: [],
          error: toDisplayText(payload.detail || "GPU check could not complete."),
        });
        return;
      }
      setPreflight(payload);
    } catch (error) {
      setPreflight({
        status: "Failed",
        gpu: null,
        recommendations: [],
        error: toDisplayText(error instanceof Error ? error.message : "GPU check could not complete."),
      });
    } finally {
      setBusy(false);
    }
  }

  async function runJob() {
    setBusy(true);
    try {
      const response = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          engines: selectedEngines,
          parameters,
          sshTarget: sshTarget || null,
        }),
      });
      const created = await readJson(response, "Run could not start.");
      if (!response.ok) {
        setJob({
          status: "failed",
          error: toDisplayText(created.detail || "Run could not start."),
          events: [
            { index: 0, type: "job_failed", message: toDisplayText(created.detail || "Run could not start.") },
          ],
          results: [],
        });
        setBusy(false);
        return;
      }
      setJob(normalizeJob(created));
      const events = new EventSource(`/api/jobs/${created.job_id}/events`);
      events.addEventListener("message", (event) => {
        const item = parseEventPayload(event.data);
        if (!item) {
          setJob((current) => failCurrentJob(current, "Job event stream sent an unreadable update."));
          setBusy(false);
          events.close();
          return;
        }
        setJob((current) => {
          if (!current) return current;
          const eventsByIndex = new Map([...eventsOf(current), item].map((entry) => [entry.index, entry]));
          return { ...current, status: statusFromEvent(item, current.status), events: Array.from(eventsByIndex.values()) };
        });
      });
      events.addEventListener("done", (event) => {
        const payload = parseJobPayload(event.data);
        setJob(payload || failCurrentJob(null, "Job finished with an unreadable result payload."));
        setBusy(false);
        events.close();
      });
      events.onerror = () => {
        setJob((current) => {
          if (!current || current.status !== "running") return current;
          return {
            ...current,
            status: "failed",
            error: "Job event stream disconnected.",
            events: [
              ...eventsOf(current),
              {
                index: eventsOf(current).length,
                type: "job_failed",
                message: "Job event stream disconnected.",
              },
            ],
          };
        });
        setBusy(false);
        events.close();
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : "Run could not start.";
      setJob({
        status: "failed",
        error: message,
        events: [{ index: 0, type: "job_failed", message }],
        results: [],
      });
      setBusy(false);
    }
  }

  async function cancelJob() {
    if (!job) return;
    const response = await fetch(`/api/jobs/${job.job_id}/cancel`, { method: "POST" });
    const payload = await readJson(response, "Cancel request failed.");
    if (!response.ok) {
      setJob((current) => failCurrentJob(current, payload.detail || "Cancel request failed."));
      return;
    }
    setJob(normalizeJob(payload));
  }

  function toggleEngine(engine) {
    setSelectedEngines((items) =>
      items.includes(engine) ? items.filter((item) => item !== engine) : [...items, engine],
    );
  }

  return (
    <div className="app-shell">
      <aside className="side-rail">
        <div className="brand">
          <span className="brand-mark">I</span>
          <span>Inferno Dashboard</span>
        </div>
        <nav>
          <a className="active"><Gauge /> Runs</a>
          <a><Cpu /> Engines</a>
          <a><ShieldCheck /> Evidence</a>
          <a><Activity /> Logs</a>
        </nav>
        <div className="rail-note">
          <span>Local only</span>
          <strong>No cloud control plane</strong>
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <h1>Inferno Dashboard</h1>
            <p>GPU-aware engine smoke tests with validated artifact output.</p>
          </div>
          <div className="topbar-actions">
            <button className="ghost" onClick={runPreflight} disabled={busy}>
              <RefreshCcw /> Check GPU
            </button>
            <button className="primary" onClick={runJob} disabled={busy || selectedEngines.length === 0}>
              <Play /> Run test
            </button>
          </div>
        </header>

        <section className="dashboard-grid">
          <section className="panel setup-panel">
            <PanelTitle icon={<RadioTower />} title="Remote host" />
            <label>
              SSH access
              <input
                value={sshTarget}
                onChange={(event) => setSshTarget(event.target.value)}
                placeholder="Use INFERNO_GPU_SSH or paste for this session"
                autoComplete="off"
              />
            </label>
            <GpuCard preflight={preflight} />
            <PanelTitle icon={<SlidersHorizontal />} title="Test parameters" />
            <label>
              Model profile
              <select
                value={parameters.modelProfile}
                onChange={(event) => setParameters({ ...parameters, modelProfile: event.target.value })}
              >
                <option value="qwen35-2b">Qwen3.5 2B</option>
              </select>
            </label>
            <label>
              KV cache
              <select
                value={parameters.kvCacheMode}
                onChange={(event) => setParameters({ ...parameters, kvCacheMode: event.target.value })}
              >
                <option value="engine_default">Engine default</option>
                <option value="compact">Compact</option>
              </select>
            </label>
            <label>
              Prompt
              <textarea
                value={parameters.prompt}
                onChange={(event) => setParameters({ ...parameters, prompt: event.target.value })}
              />
            </label>
            <div className="field-row">
              <NumberField
                label="Total requests"
                value={parameters.totalRequests}
                min="1"
                max="128"
                onChange={(value) => setParameters({ ...parameters, totalRequests: value })}
              />
              <NumberField
                label="Concurrent users"
                value={parameters.concurrentUsers}
                min="1"
                max="16"
                onChange={(value) => setParameters({ ...parameters, concurrentUsers: value })}
              />
            </div>
            <div className="field-row">
              <NumberField
                label="Request rate"
                value={parameters.requestRateRps}
                min="0.01"
                max="100"
                step="0.1"
                onChange={(value) => setParameters({ ...parameters, requestRateRps: value })}
              />
              <NumberField
                label="Warmup requests"
                value={parameters.warmupRequests}
                min="0"
                max="32"
                onChange={(value) => setParameters({ ...parameters, warmupRequests: value })}
              />
            </div>
            <div className="field-row">
              <label>
                Max tokens
                <input
                  type="number"
                  min="1"
                  max="1024"
                  value={parameters.maxTokens}
                  onChange={(event) => setParameters({ ...parameters, maxTokens: Number(event.target.value) })}
                />
              </label>
            </div>
            <div className="field-row">
              <label>
                Temperature
                <input
                  type="number"
                  min="0"
                  max="2"
                  step="0.1"
                  value={parameters.temperature}
                  onChange={(event) => setParameters({ ...parameters, temperature: Number(event.target.value) })}
                />
              </label>
              <label>
                Top-p
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.05"
                  value={parameters.topP}
                  onChange={(event) => setParameters({ ...parameters, topP: Number(event.target.value) })}
                />
              </label>
            </div>
            <label>
              Timeout seconds
              <input
                type="number"
                min="30"
                max="3600"
                value={parameters.timeoutSeconds}
                onChange={(event) => setParameters({ ...parameters, timeoutSeconds: Number(event.target.value) })}
              />
            </label>
            <label>
              Batch prompts
              <textarea
                value={parameters.batchPrompts}
                onChange={(event) => setParameters({ ...parameters, batchPrompts: event.target.value })}
                placeholder="One prompt per line. Leave empty to reuse the prompt above."
              />
            </label>
          </section>

          <section className="engines-panel">
            <div className="section-heading">
              <div>
                <h2>Engine selection</h2>
                <p>TensorRT-LLM profile changes automatically after GPU check.</p>
              </div>
              {busy ? <span className="live-chip">Running</span> : <span className="quiet-chip">Idle</span>}
            </div>
            <div className="engine-grid">
              {(catalog.engines || []).map((engine) => (
                <EngineCard
                  key={engine.id}
                  engine={engine}
                  selected={selectedEngines.includes(engine.id)}
                  recommendation={recommendations[engine.id]}
                  onToggle={() => toggleEngine(engine.id)}
                />
              ))}
            </div>
            <ProgressPanel job={job} onCancel={cancelJob} />
          </section>

          <section className="results-panel">
            <div className="section-heading">
              <div>
                <h2>Results</h2>
                <p>Validated runs, warnings, and artifact paths.</p>
              </div>
              <span className={`status-badge ${jobStatusClass(job?.status)}`}>{job?.status || "No run"}</span>
            </div>
            <ResultsTable results={job?.results || []} />
          </section>
        </section>
      </main>
    </div>
  );
}

function PanelTitle({ icon, title }) {
  return (
    <div className="panel-title">
      {icon}
      <h2>{title}</h2>
    </div>
  );
}

function NumberField({ label, value, min, max, step = "1", onChange }) {
  return (
    <label>
      {label}
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function GpuCard({ preflight }) {
  const gpu = preflight?.gpu;
  if (!preflight) {
    return (
      <div className="gpu-card muted">
        <Server />
        <div>
          <strong>Needs SSH</strong>
          <span>GPU profile not checked yet.</span>
        </div>
      </div>
    );
  }
  if (!gpu) {
    return (
      <div className="gpu-card failed">
        <AlertTriangle />
        <div>
          <strong>{preflight.status}</strong>
          <span>{preflight.error || "Remote GPU profile unavailable."}</span>
        </div>
      </div>
    );
  }
  return (
    <div className="gpu-card">
      <Cpu />
      <div>
        <strong>{gpu.name}</strong>
        <span>
          CUDA {gpu.cuda_version || "unknown"} / SM {gpu.compute_capability || "?"} / {gpu.memory_mib || "?"} MiB
        </span>
      </div>
      <span className={`status-dot ${gpu.docker_gpu_ok ? "ok" : "bad"}`} />
    </div>
  );
}

function EngineCard({ engine, selected, recommendation, onToggle }) {
  const status = recommendation?.status || "Needs SSH";
  const warnings = Array.isArray(recommendation?.warnings) ? recommendation.warnings : [];
  return (
    <button className={`engine-card ${selected ? "selected" : ""}`} onClick={onToggle}>
      <div className="engine-card-head">
        <span>{engine.label}</span>
        <span className={`status-badge ${STATUS_CLASS[status] || "is-muted"}`}>{status}</span>
      </div>
      <strong>{engine.default_model}</strong>
      <small>{engine.comparison_scope}</small>
      {recommendation?.selected_profile && (
        <span className="profile-chip">{recommendation.selected_profile}</span>
      )}
      {warnings.slice(0, 2).map((warning, index) => (
        <span className="warning-line" key={`${index}-${toDisplayText(warning)}`}>
          <AlertTriangle /> {toDisplayText(warning)}
        </span>
      ))}
    </button>
  );
}

function ProgressPanel({ job, onCancel }) {
  const events = job?.events || [];
  return (
    <div className="progress-panel">
      <div className="progress-head">
        <div>
          <h2>Progress</h2>
          <p>{job?.job_id || "No active job"}</p>
        </div>
        {job?.status === "running" && (
          <button className="danger" onClick={onCancel}>
            <StopCircle /> Cancel
          </button>
        )}
      </div>
      {job?.error && <div className="error-banner">{job.error}</div>}
      <div className="event-log">
        {events.length === 0 ? (
          <span className="empty-log">Run output will appear here.</span>
        ) : (
          events.slice(-8).map((event) => (
            <div className="event-row" key={event.index}>
              <span>{toDisplayText(event.type)}</span>
              <p>{toDisplayText(event.message)}</p>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function ResultsTable({ results }) {
  const rows = Array.isArray(results) ? results.filter(Boolean) : [];
  if (!rows.length) {
    return (
      <div className="empty-results">
        <CheckCircle2 />
        <span>Results will appear after a validated run.</span>
      </div>
    );
  }
  const e2eValues = rows.map((item) => finiteNumber(item.metrics?.e2e_ms?.p50 ?? item.summary?.e2e_ms)).filter((value) => value != null);
  const maxE2e = Math.max(...e2eValues, 1);
  return (
    <div className="results-table-wrap">
      <table>
        <thead>
          <tr>
            <th>Engine</th>
            <th>Status</th>
            <th>TTFT</th>
            <th>E2E p50/p95/p99</th>
            <th>TPS</th>
            <th>TPOT</th>
            <th>GPU</th>
            <th>VRAM</th>
            <th>Evidence</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((result, index) => {
            const metrics = result.metrics || {};
            return (
              <Fragment key={result.run_id || `${result.engine || "engine"}-${index}`}>
                <tr>
                  <td>
                    <strong>{toDisplayText(result.engine || result.requested_engine || "engine")}</strong>
                    <span>{toDisplayText(result.run_id || "run id pending")}</span>
                    {result.failure_hint && <span className="failure-hint">{toDisplayText(result.failure_hint)}</span>}
                  </td>
                  <td>
                    <span className={`status-badge ${result.validation_ok ? "is-valid" : "is-fail"}`}>
                      {result.validation_ok ? "Validated" : "Failed"}
                    </span>
                  </td>
                  <td>{formatMetric(metrics.ttft_ms?.p50 ?? result.summary?.ttft_ms, "ms")}</td>
                  <td>
                    <MetricBar value={metrics.e2e_ms?.p50 ?? result.summary?.e2e_ms} max={maxE2e} suffix="ms" />
                    <span className="result-submetric">
                      {metricTriplet(metrics.e2e_ms, "ms")}
                    </span>
                  </td>
                  <td>{formatMetric(metrics.output_tps?.value ?? result.summary?.output_token_throughput_tps, "tok/s")}</td>
                  <td>{formatMetric(metrics.tpot_ms?.p50, "ms/tok")}</td>
                  <td>{formatMetric(metrics.gpu_utilization_percent?.p95, "%")}</td>
                  <td>
                    {formatMetric(metrics.vram_used_mib?.p95, "MiB")}
                    <span className="result-submetric">
                      {formatMetric(metrics.vram_used_mib?.percent_of_total?.p95, "%")}
                    </span>
                  </td>
                  <td>
                    <ArtifactLink path={result.artifacts?.summary} label="summary" />
                    <ArtifactLink path={result.artifacts?.validation} label="validation" />
                  </td>
                </tr>
                <tr className="metric-details-row">
                  <td colSpan="9">
                    <details>
                      <summary>Details</summary>
                      <MetricDetails result={result} />
                    </details>
                  </td>
                </tr>
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function MetricDetails({ result }) {
  const metrics = result.metrics || {};
  const concurrency = metrics.concurrency || {};
  const cache = metrics.cache || {};
  return (
    <div className="metric-details-grid">
      <DetailItem label="Requests" value={requestStatus(result.summary)} />
      <DetailItem
        label="Concurrency"
        value={`cfg ${displayValue(concurrency.configured)} / max ${displayValue(concurrency.max_in_flight)} / seen ${displayValue(concurrency.observed_max_overlap)}`}
      />
      <DetailItem label="KV cache" value={efficiencyText(metrics.kv_cache_efficiency)} />
      <DetailItem label="Batching" value={efficiencyText(metrics.continuous_batching_efficiency)} />
      <DetailItem label="Scheduler" value={efficiencyText(metrics.scheduler_efficiency)} />
      <DetailItem
        label="Cache state"
        value={`${displayValue(cache.state)} / ${displayValue(cache.policy)} / warmup ${displayValue(cache.warmup_requests)} / ${displayValue(cache.kv_cache_mode)}`}
      />
      <DetailItem label="TTFT p50/p95/p99" value={metricTriplet(metrics.ttft_ms, "ms")} />
      <DetailItem label="TPOT p50/p95/p99" value={metricTriplet(metrics.tpot_ms, "ms/tok")} />
    </div>
  );
}

function DetailItem({ label, value }) {
  return (
    <div className="detail-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function MetricBar({ value, max, suffix }) {
  const number = finiteNumber(value);
  const width = number ? Math.max(8, (number / max) * 100) : 0;
  return (
    <div className="metric-bar">
      <span style={{ width: `${width}%` }} />
      <strong>{formatMetric(value, suffix)}</strong>
    </div>
  );
}

function ArtifactLink({ path, label }) {
  if (typeof path !== "string" || !path) return null;
  return (
    <a
      className="artifact-link"
      href={`/api/artifacts?path=${encodeURIComponent(path)}`}
      target="_blank"
      rel="noreferrer"
    >
      {label} <ExternalLink />
    </a>
  );
}

function formatMetric(value, suffix) {
  const number = finiteNumber(value);
  return number == null ? "n/a" : `${number.toFixed(1)} ${suffix}`;
}

function metricTriplet(metric, suffix) {
  if (!metric || metric.status !== "ok") return metric?.reason || "n/a";
  return `${formatMetric(metric.p50, suffix)} / ${formatMetric(metric.p95, suffix)} / ${formatMetric(metric.p99, suffix)}`;
}

function efficiencyText(metric) {
  if (!metric || metric.status !== "ok") return metric?.reason || "n/a";
  const suffix = metric.kind === "proxy" ? " proxy" : "";
  return `${formatMetric(metric.value, metric.unit || "%")}${suffix}`;
}

function requestStatus(summary) {
  if (!summary || summary.request_count == null) return "n/a";
  const failures = summary.error_count ? `, ${summary.error_count} failed` : "";
  return `${summary.success_count ?? 0}/${summary.request_count} ok${failures}`;
}

function displayValue(value) {
  return value == null || value === "" ? "n/a" : toDisplayText(value);
}

function jobStatusClass(status) {
  if (status === "succeeded") return "is-valid";
  if (status === "failed" || status === "cancelled") return "is-fail";
  if (status === "running") return "is-legacy";
  return "is-muted";
}

function statusFromEvent(event, fallback) {
  if (event.type === "job_started" || event.type === "engine_started" || event.type === "engine_readiness") {
    return "running";
  }
  if (event.type === "job_failed") return "failed";
  if (event.type === "job_cancelled") return "cancelled";
  if (event.type === "job_finished") return fallback === "queued" ? "running" : fallback;
  return fallback;
}

export default App;
