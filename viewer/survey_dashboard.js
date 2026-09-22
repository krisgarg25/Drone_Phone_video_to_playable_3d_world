/* Survey evidence UI. Importing this module in Node has no DOM or network effects. */
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
  const isObject = value => !!value && typeof value === "object" && !Array.isArray(value);
  const asArray = value => Array.isArray(value) ? value : [];
  const text = value => typeof value === "string" ? value : "";

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
  }
  function validScene(scene) {
    return typeof scene === "string" && /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(scene);
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
  function renderCriteria(evaluation) {
    return criteriaRows(evaluation).map(row => {
      const evidence = row.evidence;
      const detail = text(evidence?.detail) || text(evidence?.summary) || text(evidence?.reason) || row.note;
      const metrics = isObject(evidence?.metrics) ? evidence.metrics : {};
      const measured = ["measured", "meets_target", "exceeds_target", "pass", "passed", "fail", "failed"].includes(evidence?.status);
      let measurements = "";
      if (row.id === "accuracy" && measured) measurements = `<dl class="survey-evidence-metrics"><div><dt>Independent 3D RMSE</dt><dd>${formatMeasurement(metrics.rmse_3d_m, "m")}</dd></div><div><dt>Independent 3D P95</dt><dd>${formatMeasurement(metrics.p95_3d_m, "m")}</dd></div></dl>`;
      if (row.id === "speed" && measured) measurements = `<dl class="survey-evidence-metrics"><div><dt>Recorded processing</dt><dd>${formatMeasurement(metrics.elapsed_s, "s", 1)}</dd></div><div><dt>Source duration</dt><dd>${formatMeasurement(metrics.video_duration_s, "s", 1)}</dd></div></dl>`;
      return `<article class="survey-criterion">
        <div class="survey-criterion-heading"><h3>${escapeHtml(row.label)}</h3><span class="survey-weight">${row.weight}<small>% weight</small></span></div>
        <div class="survey-criterion-status">${badge(row.status)}<span>${escapeHtml(row.target)}</span></div>
        <p>${escapeHtml(detail)}</p>${measurements}
        ${evidence ? `<details><summary>Evidence and sources · ${escapeHtml(row.label)}</summary><pre>${escapeHtml(JSON.stringify(evidence, null, 2))}</pre></details>` : '<span class="survey-no-evidence">No evaluation evidence supplied</span>'}
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
  function formatMeasurement(value, unit, digits = 2) {
    return typeof value === "number" && Number.isFinite(value) && value >= 0
      ? value.toFixed(digits) + (unit ? " " + unit : "") : "Unknown";
  }
  function renderAlignment(alignment) {
    const a = isObject(alignment) ? alignment : {};
    return `<dl class="survey-metric-grid">
      <div><dt>GPS fit RMSE</dt><dd>${formatMeasurement(a.fit_rmse_m, "m")}</dd></div>
      <div><dt>Scale</dt><dd>${formatMeasurement(a.scale, "", 4)}</dd></div>
      <div><dt>Matched cameras</dt><dd>${formatMeasurement(a.matched_count, "", 0)}</dd></div>
      <div><dt>Inliers</dt><dd>${formatMeasurement(a.inlier_count, "", 0)}</dd></div>
    </dl><p class="survey-note">GPS fit is not independent accuracy. Use held-out checkpoints for the ≤ 1 m target; alignment alone does not validate it.</p>`;
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
    const announce = message => { byId("survey-message").textContent = message; };
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
      root.setAttribute("aria-busy", String(state.busy));
      byId("survey-status").textContent = state.busy ? "Working · CPU only" : data ? statusInfo(data.status).label : state.error ? "Unavailable" : "Awaiting scene";
      byId("survey-status").className = "survey-badge " + (state.error ? "bad" : data ? statusInfo(data.status).tone : "neutral");
      byId("survey-error").textContent = state.error;
      byId("survey-error").hidden = !state.error;
      byId("survey-scene-path").textContent = state.scene ? `videos/${state.scene}/` : "videos/<scene>/";
      byId("survey-checkpoint-path").textContent = state.scene ? `work/${state.scene}/survey/checkpoints.json` : "work/<scene>/survey/checkpoints.json";
      byId("survey-criteria").innerHTML = renderCriteria(data?.evaluation);
      byId("survey-alignment").innerHTML = renderAlignment(data?.alignment);
      const readiness = byId("survey-readiness");
      readiness.replaceChildren();
      for (const item of asArray(data?.readiness).filter(isObject)) {
        const row = doc.createElement("li");
        const label = doc.createElement("strong"); label.textContent = text(item.label) || "Input";
        const tag = doc.createElement("span"); const status = statusInfo(item.status);
        tag.className = "survey-badge " + status.tone; tag.textContent = status.label;
        const detail = doc.createElement("p"); detail.textContent = text(item.detail);
        row.append(label, tag, detail); readiness.appendChild(row);
      }
      if (!readiness.childElementCount) {
        const li = doc.createElement("li"); li.className = "survey-note";
        li.textContent = state.scene ? "Readiness not measured. Refresh to read scene evidence." : "Select a scene to check video, telemetry and metadata.";
        readiness.appendChild(li);
      }
      renderList("survey-blockers", asArray(data?.blockers), data ? "No blockers reported by the API. This is not an accuracy pass." : "Requirements will appear after the scene is loaded.");
      renderList("survey-warnings", [...asArray(data?.warnings), ...asArray(data?.alignment?.warnings), ...asArray(data?.evaluation?.warnings)], "No additional warnings supplied. Missing evidence remains unknown.");
      const permissions = actionAvailability(state);
      for (const name of ["save", "prepare", "align", "evaluate"]) byId("survey-" + name).disabled = !permissions[name];
      for (const id of ["survey-csv", "survey-metadata"]) byId(id).disabled = state.busy || !state.scene;
      byId("survey-refresh").disabled = state.busy;
      byId("survey-save").textContent = state.busy && state.operation === "inputs" ? "Validating and saving…" : "Validate & save inputs";
      const gpu = byId("survey-gpu-state");
      gpu.textContent = data?.gpu_execution_enabled === true ? "GPU execution reported enabled by server · not available in this lane" : "GPU permission pending · execution disabled";
      renderArtifacts(data);
      renderCommands(data);
    }
    function renderArtifacts(data) {
      const container = byId("survey-artifacts"); container.replaceChildren();
      const artifacts = asArray(data?.artifacts).filter(isObject);
      for (const artifact of artifacts) {
        const row = doc.createElement("li");
        const url = artifactUrl(artifact.url, state.scene);
        const link = doc.createElement(url ? "a" : "span");
        link.textContent = text(artifact.name) || "Evidence artifact";
        if (url) { link.href = url; link.setAttribute("download", ""); }
        const kind = doc.createElement("span"); kind.className = "survey-note";
        kind.textContent = url ? text(artifact.kind) || "Download evidence" : "Link withheld: not a safe scene-local /work/ URL";
        row.append(link, kind); container.appendChild(row);
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
  return { CRITERIA, CSV_TEMPLATE, METADATA_TEMPLATE, escapeHtml, validScene, statusInfo, criteriaRows, renderCriteria,
    artifactUrl, formatMeasurement, renderAlignment, viewFrameSource, initialState, reduceState, actionAvailability, readResponse, inputPayload, commandText, mount };
});
