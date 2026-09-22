/* Survey evidence console. Importing this module in Node has no DOM or network effects. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.SurveyDashboard = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const CRITERIA = [
    { id: "accuracy", label: "Reconstruction accuracy", weight: 30, target: "≤ 1 m independent error", note: "GPS alignment residuals are not held-out survey accuracy." },
    { id: "completeness", label: "Model completeness", weight: 20, target: "Visible single-pass scene", note: "Requires reference coverage evidence; unseen or inferred surfaces are not measured recovery." },
    { id: "speed", label: "Processing speed", weight: 20, target: "< 15 min for a 10 min video", note: "Requires a complete timed run on declared hardware, not an estimate or partial log sum." },
    { id: "innovation", label: "Innovation", weight: 15, target: "Controlled ablation", note: "Requires a baseline comparison that isolates the method's contribution." },
    { id: "scalability", label: "Scalability", weight: 10, target: "Measured resource scaling", note: "Requires comparable runs across duration, resolution or scene extent, with resource measurements." },
    { id: "ui", label: "User interface", weight: 5, target: "Import → inspect → export", note: "Interface availability is not evidence of completed operator acceptance testing." }
  ];
  // Header only: never offer synthetic positions as a real flight.
  const CSV_TEMPLATE = "t_sec,latitude_deg,longitude_deg,altitude_m,horizontal_std_m,vertical_std_m\n";
  const METADATA_TEMPLATE = JSON.stringify({ schema_version: 1, time_reference: "video", time_offset_s: 0,
    altitude_datum: "ellipsoidal", position_reference: "camera_center", single_pass: true, video_duration_s: 600 }, null, 2) + "\n";

  // Delivery formats. The checklist reports only what the artifacts list contains.
  const FORMATS = [
    { label: "OBJ", hint: "text mesh, no CRS carried", ext: ["obj"] },
    { label: "PLY", hint: "point cloud or mesh", ext: ["ply"] },
    { label: "LAS", hint: "ASPRS point exchange", ext: ["las"] },
    { label: "GeoTIFF", hint: "georeferenced raster", ext: ["tif", "tiff"] },
    { label: "glTF", hint: "runtime transmission (gltf/glb)", ext: ["gltf", "glb"] },
    { label: "FBX", hint: "DCC interchange", ext: ["fbx"] }
  ];
  // Workflow stages, driven only by the server status value.
  const STAGES = [
    { id: "inputs", code: "01", label: "Inputs" },
    { id: "prepare", code: "02", label: "Prepare" },
    { id: "align", code: "03", label: "Align" },
    { id: "evaluate", code: "04", label: "Evaluate" },
    { id: "evidence", code: "05", label: "Evidence" }
  ];
  const STAGE_REACHED = { not_prepared: 1, invalid: 2, prepared: 3, aligned: 4, evaluated: 5 };
  const MEASURED_STATES = ["measured", "meets_target", "exceeds_target", "pass", "passed", "fail", "failed"];
  const NOT_AVAILABLE = "not available";

  const isObject = value => !!value && typeof value === "object" && !Array.isArray(value);
  const asArray = value => Array.isArray(value) ? value : [];
  const text = value => typeof value === "string" ? value : "";

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
  }
  function validScene(scene) {
    return typeof scene === "string" && /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(scene);
  }
  // First non-empty string among candidate server fields, else "".
  function firstText(...values) {
    for (const value of values) if (typeof value === "string" && value.trim()) return value.trim();
    return "";
  }
  function statusInfo(value) {
    const statuses = {
      unknown: ["Unknown", "neutral"], missing: ["Missing", "warn"], invalid: ["Invalid", "bad"],
      ready: ["Ready", "good"], measured: ["Measured", "info"], pass: ["Passed", "good"], passed: ["Passed", "good"],
      fail: ["Failed", "bad"], failed: ["Failed", "bad"], partial: ["Partial", "warn"],
      pending: ["Pending", "warn"], blocked: ["Blocked", "warn"], not_evaluated: ["Not evaluated", "neutral"],
      not_prepared: ["Not prepared", "neutral"], prepared: ["Prepared", "info"], aligned: ["Aligned", "info"],
      evaluated: ["Evaluated", "info"], not_demonstrated: ["Not demonstrated", "neutral"],
      meets_target: ["Meets target", "good"], exceeds_target: ["Exceeds target", "bad"]
    };
    const pair = Object.prototype.hasOwnProperty.call(statuses, value) ? statuses[value] : statuses.unknown;
    return { label: pair[0], tone: pair[1] };
  }
  // Ledger row state: drives the stripe, the fill mark and the hatch. Never colour alone.
  function stateCode(status) {
    switch (status) {
      case "measured": case "pass": case "passed": return "measured";
      case "meets_target": return "meets";
      case "exceeds_target": return "over";
      case "fail": case "failed": return "failed";
      case "not_evaluated": return "not-evaluated";
      case "not_demonstrated": return "not-demonstrated";
      case "missing": case "invalid": return "missing";
      case "pending": case "blocked": case "partial": return "pending";
      default: return "unknown";
    }
  }
  function badge(status) {
    return `<span class="survey-badge ${status.tone}">${escapeHtml(status.label)}</span>`;
  }
  function criterionId(item) {
    const id = text(item.id).toLowerCase();
    const aliases = { reconstruction_accuracy: "accuracy", model_completeness: "completeness", processing_speed: "speed", user_interface: "ui", interface: "ui" };
    if (CRITERIA.some(c => c.id === id)) return id;
    if (aliases[id]) return aliases[id];
    const label = text(item.label).toLowerCase();
    return CRITERIA.find(c => label === c.label.toLowerCase())?.id;
  }
  function criteriaRows(evaluation) {
    const items = asArray(evaluation?.criteria).filter(isObject);
    return CRITERIA.map(criterion => {
      const evidence = items.find(item => criterionId(item) === criterion.id);
      return { ...criterion, evidence: evidence || null, status: statusInfo(evidence?.status) };
    });
  }
  function formatMeasurement(value, unit, digits = 2) {
    return typeof value === "number" && Number.isFinite(value) && value >= 0
      ? value.toFixed(digits) + (unit ? " " + unit : "") : "Unknown";
  }
  // Console-wide token for an absent number. formatMeasurement keeps its contract.
  function readNumber(value, unit, digits) {
    const rendered = formatMeasurement(value, unit, digits);
    return rendered === "Unknown" ? NOT_AVAILABLE : rendered;
  }
  function metricsOf(evidence) {
    return isObject(evidence?.metrics) ? evidence.metrics : {};
  }
  function humanKey(key) {
    return String(key).replace(/_/g, " ").replace(/\b\w/g, ch => ch.toUpperCase());
  }
  function evidenceDetail(row) {
    return text(row.evidence?.detail) || text(row.evidence?.summary) || text(row.evidence?.reason) || row.note;
  }
  function measuredValue(row) {
    if (!MEASURED_STATES.includes(row.evidence?.status)) {
      return `<span class="survey-val-none">${NOT_AVAILABLE}</span>`;
    }
    const metrics = metricsOf(row.evidence);
    if (row.id === "accuracy") {
      return `<span class="survey-val">${escapeHtml(readNumber(metrics.rmse_3d_m, "m"))}</span>
        <span class="survey-val-sub">3D RMSE · P95 ${escapeHtml(readNumber(metrics.p95_3d_m, "m"))}</span>`;
    }
    if (row.id === "speed") {
      return `<span class="survey-val">${escapeHtml(readNumber(metrics.elapsed_s, "s", 1))}</span>
        <span class="survey-val-sub">for ${escapeHtml(readNumber(metrics.video_duration_s, "s", 0))} of video</span>`;
    }
    const numeric = Object.entries(metrics).filter(([, v]) => typeof v === "number" && Number.isFinite(v)).slice(0, 2);
    if (!numeric.length) return `<span class="survey-val-none">measured · no numeric statistic returned</span>`;
    return numeric.map(([key, value]) => `<span class="survey-val">${escapeHtml(readNumber(value, ""))}</span>
      <span class="survey-val-sub">${escapeHtml(humanKey(key))}</span>`).join("");
  }
  function renderCriteria(evaluation) {
    return criteriaRows(evaluation).map(row => {
      const state = stateCode(row.evidence?.status);
      const ticks = Math.round(row.weight / 5);
      const detail = evidenceDetail(row);
      return `<article class="survey-criterion" data-state="${state}" data-criterion="${row.id}">
        <div class="survey-cell-weight"><span class="survey-weight">${row.weight}</span>
          <span class="survey-ticks" data-ticks="${ticks}" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i></span>
          <span class="survey-vh">per cent of official weighting</span></div>
        <div class="survey-cell-name"><h3>${escapeHtml(row.label)}</h3>
          <p class="survey-row-detail">${escapeHtml(detail)}</p></div>
        <div class="survey-cell-target"><span class="survey-k">Official target</span>
          <span class="survey-t">${escapeHtml(row.target)}</span></div>
        <div class="survey-cell-status"><span class="survey-mark" aria-hidden="true"></span>
          <span class="survey-state">${escapeHtml(row.status.label)}</span></div>
        <div class="survey-cell-value"><span class="survey-k">Measured</span>${measuredValue(row)}</div>
        ${row.evidence ? `<details class="survey-row-audit"><summary>Evidence and sources · ${escapeHtml(row.label)}</summary><pre>${escapeHtml(JSON.stringify(row.evidence, null, 2))}</pre></details>`
          : `<p class="survey-row-audit survey-no-evidence">No evaluation evidence supplied for this criterion.</p>`}
      </article>`;
    }).join("");
  }
  function artifactUrl(value, scene) {
    if (typeof value !== "string" || !validScene(scene) || /[\\?#\s]/.test(value) || !value.startsWith(`/work/${scene}/`)) return null;
    let decoded;
    try { decoded = decodeURIComponent(value); } catch (_) { return null; }
    if (/%|[\\?#\x00-\x1f\x7f]/.test(decoded) || /%2f/i.test(value)) return null;
    const segments = decoded.split("/").slice(1);
    if (!segments.every(s => s && s !== "." && s !== ".." && /^[A-Za-z0-9_. -]+$/.test(s))) return null;
    return value;
  }
  function renderAlignment(alignment) {
    const a = isObject(alignment) ? alignment : {};
    return `<div class="survey-readout" data-tone="fit"><span class="survey-ro-k">GPS fit RMSE</span>
      <span class="survey-big">${escapeHtml(readNumber(a.fit_rmse_m, "m"))}</span>
      <span class="survey-sub">residual of the alignment fit</span></div>
      <dl class="survey-inline-readouts">
        <div><dt>Scale</dt><dd>${escapeHtml(readNumber(a.scale, "", 4))}</dd></div>
        <div><dt>Matched cameras</dt><dd>${escapeHtml(readNumber(a.matched_count, "", 0))}</dd></div>
        <div><dt>Inliers</dt><dd>${escapeHtml(readNumber(a.inlier_count, "", 0))}</dd></div>
      </dl>
      <p class="survey-note">GPS fit is not independent accuracy. Use held-out checkpoints for the ≤ 1 m target; alignment alone does not validate it.</p>`;
  }
  function accuracyRow(evaluation) {
    return criteriaRows(evaluation).find(row => row.id === "accuracy");
  }
  function renderAccuracy(evaluation) {
    const row = accuracyRow(evaluation);
    const measured = MEASURED_STATES.includes(row?.evidence?.status);
    const metrics = metricsOf(row?.evidence);
    const counts = isObject(evaluation?.counts) ? evaluation.counts : {};
    return `<div class="survey-readout" data-tone="${measured ? "held" : "gap"}"><span class="survey-ro-k">Held-out 3D accuracy</span>
      <span class="survey-big">${escapeHtml(measured ? readNumber(metrics.rmse_3d_m, "m") : NOT_AVAILABLE)}</span>
      <span class="survey-sub">independent checkpoints · ${escapeHtml(statusInfo(row?.evidence?.status).label)}</span></div>
      <dl class="survey-inline-readouts">
        <div><dt>P95</dt><dd>${escapeHtml(measured ? readNumber(metrics.p95_3d_m, "m") : NOT_AVAILABLE)}</dd></div>
        <div><dt>Target</dt><dd>≤ 1 m</dd></div>
        <div><dt>Checkpoints</dt><dd>${escapeHtml(Number.isFinite(counts.checkpoints) ? String(counts.checkpoints) : NOT_AVAILABLE)}</dd></div>
      </dl>
      <p class="survey-note">This number and the GPS fit residual are separate measurements. Only the held-out value counts against the accuracy target.</p>`;
  }
  function renderRun(data) {
    const run = isObject(data?.latest_run) ? data.latest_run : {};
    const steps = asArray(run.steps).filter(isObject);
    const runState = ["complete", "failed", "running", "partial"].includes(text(run.status)) ? text(run.status) : NOT_AVAILABLE;
    return `<div class="survey-readout" data-tone="run"><span class="survey-ro-k">Run total</span>
      <span class="survey-big">${escapeHtml(readNumber(run.secs, "s", 1))}</span>
      <span class="survey-sub">latest recorded run · ${escapeHtml(runState)}</span></div>
      <dl class="survey-inline-readouts">
        <div><dt>Stages timed</dt><dd>${steps.length ? steps.length : NOT_AVAILABLE}</dd></div>
        <div><dt>Run id</dt><dd>${escapeHtml(firstText(run.id) || NOT_AVAILABLE)}</dd></div>
      </dl>
      <p class="survey-note">Timings come from the run record the server published. No estimate is shown in their place.</p>`;
  }
  function renderStageTimings(data) {
    const steps = asArray(isObject(data?.latest_run) ? data.latest_run.steps : []);
    const timed = steps.filter(step => isObject(step) && typeof step.secs === "number" && Number.isFinite(step.secs));
    if (!timed.length) {
      return `<p class="survey-empty">Stage timings not available. The server has not published a per-stage breakdown for the latest run.</p>`;
    }
    const slowest = Math.max(...timed.map(step => step.secs), 1);
    return `<table class="survey-timetable"><caption class="survey-vh">Stage timings for the latest recorded run</caption>
      <thead><tr><th scope="col">Stage</th><th scope="col">Status</th><th scope="col" class="survey-share">Share</th><th scope="col" class="survey-num">Seconds</th></tr></thead>
      <tbody>${timed.map(step => `<tr><td>${escapeHtml(firstText(step.name, step.stage) || "unnamed stage")}</td>
        <td>${escapeHtml(firstText(step.status) || NOT_AVAILABLE)}</td>
        <td class="survey-share"><span class="survey-share-bar" aria-hidden="true"><i style="width:${Math.max(2, Math.round(step.secs / slowest * 100))}px"></i></span></td>
        <td class="survey-num">${escapeHtml(step.secs.toFixed(1))}</td></tr>`).join("")}</tbody></table>`;
  }
  function renderInstrument(data, scene) {
    const d = isObject(data) ? data : {};
    const a = isObject(d.alignment) ? d.alignment : {};
    const crs = firstText(d.crs, d.coordinate_reference_system, d.georeference?.crs, d.evaluation?.crs, a.coordinate_frame);
    const datum = firstText(d.vertical_datum, d.datum?.vertical, d.georeference?.vertical_datum,
      d.evaluation?.vertical_datum, d.metadata?.altitude_datum);
    const reference = firstText(d.position_reference, d.metadata?.position_reference, d.georeference?.position_reference);
    return `<div class="survey-readouts" data-group="identity">
        <div class="survey-readout"><span class="survey-ro-k">Scene</span><span class="survey-ro-v survey-mono">${escapeHtml(validScene(scene) ? scene : "none selected")}</span></div>
        <div class="survey-readout"><span class="survey-ro-k">CRS / horizontal frame</span><span class="survey-ro-v">${escapeHtml(crs || NOT_AVAILABLE)}</span></div>
        <div class="survey-readout"><span class="survey-ro-k">Vertical datum</span><span class="survey-ro-v">${escapeHtml(datum || NOT_AVAILABLE)}</span></div>
        <div class="survey-readout"><span class="survey-ro-k">Position reference</span><span class="survey-ro-v">${escapeHtml(reference || NOT_AVAILABLE)}</span></div>
      </div>
      <div class="survey-readouts" data-group="fit">${renderAlignment(a)}</div>
      <div class="survey-readouts" data-group="held">${renderAccuracy(d.evaluation)}</div>
      <div class="survey-readouts" data-group="run">${renderRun(d)}</div>`;
  }
  function renderStages(status) {
    const reached = Number.isInteger(STAGE_REACHED[status]) ? STAGE_REACHED[status] : 0;
    return STAGES.map((stage, index) => {
      const state = !reached ? "idle" : index + 1 < reached ? "done" : index + 1 === reached ? "current" : "next";
      const marker = { idle: "not reached", done: "reached", current: "current", next: "not reached" }[state];
      return `<li class="survey-stage" data-state="${state}"><span class="survey-stage-code">${stage.code}</span>
        <span class="survey-stage-name">${escapeHtml(stage.label)}</span>
        <span class="survey-stage-state">${marker}</span></li>`;
    }).join("");
  }
  function fileExtension(artifact) {
    const source = firstText(artifact.name, artifact.url);
    const base = source.split("/").pop() || "";
    const match = /\.([A-Za-z0-9]{1,8})$/.exec(base);
    return match ? match[1].toLowerCase() : "";
  }
  function formatsReport(artifacts) {
    const list = asArray(artifacts).filter(isObject);
    return FORMATS.map(format => ({
      ...format,
      files: list.filter(artifact => format.ext.includes(fileExtension(artifact)))
        .map(artifact => firstText(artifact.name) || firstText(artifact.url))
    }));
  }
  function renderFormats(artifacts) {
    return formatsReport(artifacts).map(format => {
      const delivered = format.files.length > 0;
      return `<li class="survey-check" data-state="${delivered ? "delivered" : "absent"}">
        <span class="survey-check-box" aria-hidden="true"></span>
        <span class="survey-check-name">${escapeHtml(format.label)}</span>
        <span class="survey-check-hint">${escapeHtml(format.hint)}</span>
        <span class="survey-check-state">${delivered ? `delivered · ${format.files.length}` : "not delivered"}</span>
      </li>`;
    }).join("");
  }
  function measurementValue(item) {
    if (typeof item.value === "number") return readNumber(item.value, text(item.unit));
    return firstText(item.value, item.display) || NOT_AVAILABLE;
  }
  function renderMeasurements(data) {
    const rows = asArray(data?.measurements).filter(isObject);
    if (!rows.length) {
      return `<p class="survey-empty">No measurements exist yet. This section renders only what the survey API returns as
        measurements for this scene; nothing here is estimated, averaged in from another scene, or carried over from an older run.</p>
        <p class="survey-note">Measurements appear when Evaluate returns coverage, surface-reference or resource statistics.
        Until then the criteria that depend on them stay marked not evaluated.</p>`;
    }
    return `<table class="survey-table"><caption class="survey-vh">Measurements reported by the survey API</caption>
      <thead><tr><th scope="col">Measurement</th><th scope="col" class="survey-num">Value</th><th scope="col">Method</th><th scope="col">Source</th></tr></thead>
      <tbody>${rows.map(item => `<tr><td>${escapeHtml(firstText(item.label, item.name) || "unnamed measurement")}</td>
        <td class="survey-num">${escapeHtml(measurementValue(item))}</td>
        <td>${escapeHtml(firstText(item.method, item.source_method) || NOT_AVAILABLE)}</td>
        <td>${escapeHtml(firstText(item.source, item.artifact) || NOT_AVAILABLE)}</td></tr>`).join("")}</tbody></table>`;
  }
  function viewFrameSource(lane, source) {
    return lane === "view" && source ? source : "about:blank";
  }
  function initialState(scene = "") {
    return { scene, requestId: null, operation: "", busy: false, data: null, error: "" };
  }
  function reduceState(state, action) {
    if (action.type === "select") return initialState(action.scene);
    if (action.type === "begin") return { ...state, requestId: action.id, busy: true, operation: action.operation, error: "" };
    if (action.id !== state.requestId || state.requestId === null) return state;
    if (action.type === "reject") return { ...state, busy: false, operation: "", data: null, error: String(action.error || "Request failed") };
    if (action.type === "resolve") {
      const data = action.data;
      let error = "";
      if (!isObject(data) || data.schema_version !== 1) error = "Unsupported survey API schema. Refresh after the server is updated.";
      else if (data.scene !== state.scene) error = "Survey response scene does not match the selected scene. Refresh to retry.";
      else if (!["not_prepared", "prepared", "aligned", "evaluated", "invalid"].includes(data.status)) error = "Unknown survey state. Refresh after checking the server.";
      return { ...state, busy: false, operation: "", data: error ? null : data, error };
    }
    return state;
  }
  function actionAvailability(state) {
    const available = validScene(state.scene) && !!state.data && !state.busy && !state.error;
    const status = state.data?.status;
    return {
      save: available && ["not_prepared", "invalid"].includes(status),
      prepare: available && ["not_prepared", "invalid"].includes(status),
      align: available && ["prepared", "aligned", "evaluated"].includes(status),
      evaluate: available && ["prepared", "aligned", "evaluated"].includes(status)
    };
  }
  async function readResponse(response) {
    let data;
    try { data = await response.json(); }
    catch (_) { throw new Error(response.ok ? "Server returned invalid JSON. Check the survey API and refresh." : `HTTP ${response.status}: Survey API unavailable. Check the server and refresh.`); }
    if (!response.ok || (isObject(data) && data.error)) throw new Error(`HTTP ${response.status}: ${text(data?.error) || "Survey request failed"}`);
    return data;
  }
  function inputPayload(scene, csv, metadataText) {
    if (!validScene(scene)) throw new Error("Select a safe scene name (1–64 letters, numbers, underscores or hyphens).");
    if (typeof csv !== "string" || !csv.trim()) throw new Error("Select a non-empty telemetry CSV.");
    let metadata;
    try { metadata = JSON.parse(metadataText); } catch (_) { throw new Error("Flight metadata must be valid JSON."); }
    if (!isObject(metadata)) throw new Error("Flight metadata must be a JSON object.");
    const payload = { scene, telemetry_csv: csv, metadata };
    if (new TextEncoder().encode(JSON.stringify(payload)).length > 2 * 1024 * 1024) throw new Error("Inputs exceed the 2 MB JSON request limit.");
    return payload;
  }
  function commandText(command) {
    if (!Array.isArray(command?.argv) || !command.argv.length || !command.argv.every(arg => typeof arg === "string" && !/[\x00-\x1f\x7f]/.test(arg))) return "Command unavailable";
    // POSIX shell display only, never executed by this UI. argv JSON is also shown.
    return command.argv.map(arg => /^[A-Za-z0-9_./:=+-]+$/.test(arg) ? arg : "'" + arg.replace(/'/g, "'\\''") + "'").join(" ");
  }

  function mount(root, options = {}) {
    const doc = root.ownerDocument;
    const byId = id => doc.getElementById(id);
    let state = initialState();
    let active = false;
    let sequence = 0;
    let controller = null;
    let sceneList = [];
    let pulseTimer = null;
    let lastSignature = "";
    const announce = message => { byId("survey-message").textContent = message; };
    // One restrained indication that the readouts were replaced. Skipped when the
    // operator asked for reduced motion, and it never changes layout.
    function pulse() {
      const view = doc.defaultView;
      const motion = view && typeof view.matchMedia === "function" ? view.matchMedia("(prefers-reduced-motion: reduce)") : null;
      if (motion && motion.matches) return;
      root.classList.remove("survey-updated");
      void root.offsetWidth;
      root.classList.add("survey-updated");
      clearTimeout(pulseTimer);
      pulseTimer = setTimeout(() => root.classList.remove("survey-updated"), 900);
    }
    const renderList = (id, values, empty) => {
      const element = byId(id);
      element.replaceChildren();
      for (const value of values) {
        const li = doc.createElement("li");
        li.textContent = typeof value === "string" ? value : JSON.stringify(value);
        element.appendChild(li);
      }
      if (!values.length) {
        const li = doc.createElement("li"); li.className = "survey-note"; li.textContent = empty; element.appendChild(li);
      }
    };
    function render() {
      const data = state.data;
      const signature = JSON.stringify([state.scene, state.error, data]);
      root.setAttribute("aria-busy", String(state.busy));
      const status = byId("survey-status");
      status.textContent = state.busy ? "Working · CPU only" : data ? statusInfo(data.status).label : state.error ? "Unavailable" : "Awaiting scene";
      status.className = "survey-badge " + (state.error ? "bad" : data ? statusInfo(data.status).tone : "neutral");
      byId("survey-error").textContent = state.error;
      byId("survey-error").hidden = !state.error;
      byId("survey-scene-name").textContent = state.scene || "no scene selected";
      byId("survey-stages").innerHTML = renderStages(data?.status);
      byId("survey-strip").innerHTML = renderInstrument(data, state.scene);
      byId("survey-timings").innerHTML = renderStageTimings(data);
      byId("survey-scene-path").textContent = state.scene ? `videos/${state.scene}/` : "videos/<scene>/";
      byId("survey-checkpoint-path").textContent = state.scene ? `work/${state.scene}/survey/checkpoints.json` : "work/<scene>/survey/checkpoints.json";
      byId("survey-criteria").innerHTML = renderCriteria(data?.evaluation);
      byId("survey-formats").innerHTML = renderFormats(data?.artifacts);
      byId("survey-measurements").innerHTML = renderMeasurements(data);
      const readiness = byId("survey-readiness");
      readiness.replaceChildren();
      for (const item of asArray(data?.readiness).filter(isObject)) {
        const row = doc.createElement("li");
        row.className = "survey-ready-row";
        row.dataset.state = stateCode(item.status);
        const label = doc.createElement("strong"); label.textContent = text(item.label) || "Input";
        const tag = doc.createElement("span"); const status2 = statusInfo(item.status);
        tag.className = "survey-badge " + status2.tone; tag.textContent = status2.label;
        const detail = doc.createElement("p"); detail.textContent = text(item.detail);
        row.append(label, tag, detail); readiness.appendChild(row);
      }
      if (!readiness.childElementCount) {
        const li = doc.createElement("li"); li.className = "survey-note";
        li.textContent = state.scene ? "Readiness not measured. Refresh to read scene evidence." : "Select a scene to check video, telemetry and metadata.";
        readiness.appendChild(li);
      }
      const blockers = asArray(data?.blockers);
      renderList("survey-blockers", blockers, data ? "No blockers reported by the API. This is not an accuracy pass." : "Requirements will appear after the scene is loaded.");
      byId("survey-blockers-panel").classList.toggle("is-clear", blockers.length === 0);
      // What the evaluation itself refuses to prove is a limitation, so it joins
      // the provenance list rather than being dropped.
      renderList("survey-warnings", [...asArray(data?.warnings), ...asArray(data?.alignment?.warnings),
        ...asArray(data?.evaluation?.warnings), ...asArray(data?.evaluation?.what_this_does_not_prove)],
        "No additional warnings supplied. Missing evidence remains unknown.");
      const permissions = actionAvailability(state);
      for (const name of ["save", "prepare", "align", "evaluate"]) byId("survey-" + name).disabled = !permissions[name];
      for (const id of ["survey-csv", "survey-metadata"]) byId(id).disabled = state.busy || !state.scene;
      byId("survey-refresh").disabled = state.busy;
      byId("survey-save").textContent = state.busy && state.operation === "inputs" ? "Validating and saving…" : "Validate & save inputs";
      const gpu = byId("survey-gpu-state");
      gpu.textContent = data?.gpu_execution_enabled === true ? "GPU execution reported enabled by server · not available in this lane" : "GPU permission pending · execution disabled";
      renderArtifacts(data);
      renderCommands(data);
      // Only a changed readout set pulses the strip; a busy toggle must not.
      if (signature !== lastSignature && (data || state.error)) pulse();
      lastSignature = signature;
    }
    function renderArtifacts(data) {
      const container = byId("survey-artifacts"); container.replaceChildren();
      const artifacts = asArray(data?.artifacts).filter(isObject);
      for (const artifact of artifacts) {
        const row = doc.createElement("li");
        row.className = "survey-evidence-row";
        const url = artifactUrl(artifact.url, state.scene);
        const link = doc.createElement(url ? "a" : "span");
        link.textContent = text(artifact.name) || "Evidence artifact";
        if (url) { link.href = url; link.setAttribute("download", ""); }
        const kind = doc.createElement("span"); kind.className = "survey-note";
        kind.textContent = url ? text(artifact.kind) || "Download evidence" : "Link withheld: not a safe scene-local /work/ URL";
        const path = doc.createElement("span"); path.className = "survey-evidence-path";
        path.textContent = url || "no scene-local path";
        row.append(link, kind, path); container.appendChild(row);
      }
      if (!artifacts.length) {
        const li = doc.createElement("li"); li.className = "survey-note";
        li.textContent = "No evidence artifacts reported. Prepare and evaluate this scene to create available reports.";
        container.appendChild(li);
      }
    }
    function renderCommands(data) {
      const container = byId("survey-commands"); container.replaceChildren();
      for (const command of asArray(data?.commands).filter(isObject)) {
        const item = doc.createElement("div"); item.className = "survey-command";
        const label = doc.createElement("strong"); label.textContent = text(command.label) || "Suggested command";
        const note = doc.createElement("p"); note.className = "survey-note";
        note.textContent = command.requires_gpu === false ? "CPU command · display only" : "Uses GPU or permission unknown · display only; explicit permission required";
        const pre = doc.createElement("pre"); pre.textContent = commandText(command);
        const copy = doc.createElement("button"); copy.type = "button"; copy.className = "btn-ghost"; copy.textContent = "Copy command";
        copy.setAttribute("aria-label", "Copy command: " + label.textContent);
        copy.disabled = state.busy || pre.textContent === "Command unavailable";
        copy.addEventListener("click", async () => {
          try { await navigator.clipboard.writeText(pre.textContent); announce("Command copied. Nothing was executed."); }
          catch (_) { announce("Clipboard unavailable. Select and copy the command text manually."); }
        });
        const details = doc.createElement("details");
        const summary = doc.createElement("summary"); summary.textContent = "Exact argument array (check your shell before use)";
        const argv = doc.createElement("pre"); argv.textContent = JSON.stringify(command.argv, null, 2);
        details.append(summary, argv); item.append(label, note, pre, copy, details); container.appendChild(item);
      }
      if (!container.childElementCount) container.textContent = "No commands supplied. This lane never starts a reconstruction.";
    }
    async function request(operation, makePayload) {
      if (!validScene(state.scene) || state.busy) return;
      const scene = state.scene;
      const id = ++sequence;
      if (controller) controller.abort();
      const ownController = new AbortController(); controller = ownController;
      state = reduceState(state, { type: "begin", id, operation }); render();
      announce(operation === "load" ? `Loading survey evidence for ${scene}…` : `${operation === "inputs" ? "Validating inputs" : operation === "prepare" ? "Preparing survey" : operation === "align" ? "Aligning existing reconstruction" : "Evaluating evidence"} for ${scene} · CPU only…`);
      const timeout = setTimeout(() => ownController.abort(), operation === "load" ? 20000 : 120000);
      try {
        const payload = makePayload ? await makePayload(scene) : { scene };
        // File reading can finish after a scene switch. Never send it to the new scene.
        if (state.requestId !== id) return;
        if (ownController.signal.aborted) throw new DOMException("Request timed out", "AbortError");
        let data = await readResponse(await fetch(operation === "load" ? `/api/survey?scene=${encodeURIComponent(scene)}` : `/api/survey/${operation}`, {
          method: operation === "load" ? "GET" : "POST", signal: ownController.signal, cache: "no-store",
          ...(operation === "load" ? {} : { headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) })
        }));
        if (operation !== "load" && (!isObject(data) || data.schema_version === undefined)) {
          data = await readResponse(await fetch(`/api/survey?scene=${encodeURIComponent(scene)}`, { signal: ownController.signal, cache: "no-store" }));
        }
        if (state.requestId !== id) return;
        state = reduceState(state, { type: "resolve", id, data });
        if (!state.error && operation === "inputs") byId("survey-input-form").reset();
        announce(state.error ? "Survey response could not be used." : operation === "load" ? `Evidence loaded for ${scene}.` : `CPU operation complete for ${scene}. Evidence refreshed.`);
      } catch (error) {
        if (state.requestId !== id) return;
        const message = error.name === "AbortError" ? "Request timed out. The CPU operation may still finish on the server; refresh before retrying." : error instanceof TypeError ? "Survey API unreachable. Check the server connection, then refresh." : error.message;
        state = reduceState(state, { type: "reject", id, error: message }); announce("Survey request failed. See the error for the next step.");
      } finally {
        clearTimeout(timeout);
        if (state.requestId === id) { controller = null; render(); }
      }
    }
    function setScene(scene) {
      if (scene === state.scene) return;
      const wasBusy = state.busy;
      if (controller) controller.abort();
      controller = null;
      state = reduceState(state, { type: "select", scene });
      byId("survey-input-form").reset();
      syncSceneOptions(); render();
      if (active && scene) request("load");
      else announce(wasBusy ? "Scene changed. Any submitted CPU operation may still finish for the previous scene." : "Select a scene to begin.");
    }
    function syncSceneOptions() {
      const select = byId("survey-scene"); select.replaceChildren();
      const placeholder = doc.createElement("option"); placeholder.value = "";
      placeholder.textContent = sceneList.length ? "Select a scene" : "No scenes found — add a source video"; select.appendChild(placeholder);
      const names = [...new Set([...sceneList, ...(state.scene ? [state.scene] : [])])];
      for (const name of names) { const option = doc.createElement("option"); option.value = name; option.textContent = name; select.appendChild(option); }
      select.value = state.scene;
    }
    byId("survey-scene").addEventListener("change", event => {
      if (options.onSceneChange) options.onSceneChange(event.target.value);
      else setScene(event.target.value);
    });
    byId("survey-refresh").addEventListener("click", async () => {
      if (options.refreshScenes) await options.refreshScenes();
      request("load");
    });
    byId("survey-input-form").addEventListener("submit", event => {
      event.preventDefault();
      if (!actionAvailability(state).save) return;
      const csv = byId("survey-csv").files[0]; const metadata = byId("survey-metadata").files[0];
      request("inputs", async scene => {
        if (!csv || !metadata) throw new Error("Choose both a telemetry CSV and a flight metadata JSON file.");
        if (csv.size + metadata.size > 2 * 1024 * 1024) throw new Error("Inputs exceed the 2 MB request limit.");
        return inputPayload(scene, await csv.text(), await metadata.text());
      });
    });
    for (const operation of ["prepare", "align", "evaluate"]) byId("survey-" + operation).addEventListener("click", () => {
      if (actionAvailability(state)[operation]) request(operation);
    });
    for (const [id, filename, content, type] of [
      ["survey-csv-template", "telemetry-template.csv", CSV_TEMPLATE, "text/csv"],
      ["survey-metadata-template", "flight-metadata-template.json", METADATA_TEMPLATE, "application/json"]
    ]) byId(id).addEventListener("click", () => {
      const url = URL.createObjectURL(new Blob([content], { type }));
      const link = doc.createElement("a"); link.href = url; link.download = filename; doc.body.appendChild(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      announce("Template downloaded. Replace or verify every value before saving real flight inputs.");
    });
    byId("survey-csv-header").textContent = CSV_TEMPLATE.trim();
    byId("survey-metadata-example").textContent = METADATA_TEMPLATE;
    render();
    return {
      setScene,
      setScenes(items) {
        sceneList = asArray(items).map(item => text(item?.name)).filter(validScene); syncSceneOptions();
        if (!state.scene) { byId("survey-error").textContent = ""; byId("survey-error").hidden = true; }
      },
      setActive(value) { active = !!value; if (active && state.scene && !state.data && !state.busy) request("load"); },
      showSceneError(message) { if (!state.scene) { byId("survey-error").textContent = message; byId("survey-error").hidden = false; } },
      refresh() { if (active && !state.busy) request("load"); }
    };
  }
  return { CRITERIA, CSV_TEMPLATE, METADATA_TEMPLATE, FORMATS, STAGES, escapeHtml, validScene, firstText, statusInfo,
    stateCode, criteriaRows, renderCriteria, artifactUrl, formatMeasurement, readNumber, renderAlignment, renderAccuracy,
    renderRun, renderStageTimings, renderInstrument, renderStages, renderFormats, formatsReport, renderMeasurements,
    viewFrameSource, initialState, reduceState, actionAvailability, readResponse, inputPayload, commandText, mount };
});
