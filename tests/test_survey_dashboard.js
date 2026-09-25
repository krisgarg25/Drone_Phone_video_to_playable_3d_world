"use strict";
// Pure production render/state tests. No browser, jsdom, network, or GPU.
const test = require("node:test");
const assert = require("node:assert/strict");
const Survey = require("../viewer/survey_dashboard.js");

const response = (extra = {}) => ({
  schema_version: 1, scene: "flight_01", status: "not_prepared",
  readiness: [], blockers: [], evaluation: null, alignment: null,
  artifacts: [], commands: [], gpu_execution_enabled: false, ...extra
});

test("empty evidence renders all six real weights without fabricated passes or totals", () => {
  const rows = Survey.criteriaRows(null);
  assert.deepEqual(rows.map(r => r.weight), [30, 20, 20, 15, 10, 5]);
  assert.ok(rows.every(r => r.status.label === "Unknown"));
  const html = Survey.renderCriteria(null);
  assert.equal((html.match(/class="survey-criterion"/g) || []).length, 6);
  assert.match(html, /Reconstruction accuracy/);
  assert.match(html, /User interface/);
  assert.doesNotMatch(html, /Passed|overall score|100%/i);
});

test("explicit measured, failed and unsupported server statuses remain distinct", () => {
  assert.deepEqual(Survey.statusInfo("measured"), { label: "Measured", tone: "info" });
  assert.deepEqual(Survey.statusInfo("fail"), { label: "Failed", tone: "bad" });
  assert.deepEqual(Survey.statusInfo("pass"), { label: "Passed", tone: "good" });
  assert.equal(Survey.statusInfo("ready").label, "Ready");
  assert.equal(Survey.statusInfo("invented_pass").label, "Unknown");
  assert.equal(Survey.statusInfo(null).label, "Unknown");
});

test("criteria preserve server evidence and cannot turn absent status into a pass", () => {
  const evaluation = { criteria: [
    { id: "accuracy", label: "Survey accuracy", weight: 30, fit_rmse_m: 0.12 },
    { id: "processing_speed", label: "Processing Speed", weight: 20, status: "fail", detail: "Timing exceeds limit" }
  ] };
  const rows = Survey.criteriaRows(evaluation);
  assert.equal(rows[0].status.label, "Unknown");
  assert.equal(rows[2].status.label, "Failed");
  assert.match(Survey.renderCriteria(evaluation), /Timing exceeds limit/);
});

test("server strings cannot become markup or escape attributes", () => {
  const html = Survey.renderCriteria({ criteria: [{
    id: "accuracy", label: '<img src=x onerror="alert(1)">', weight: 30,
    status: "measured", detail: '<script>alert("x")</script>',
    evidence_sources: ['<a href="javascript:evil()">source</a>']
  }] });
  assert.doesNotMatch(html, /<script>|<img |<a href="javascript:/);
  assert.match(html, /&lt;script&gt;/);
  assert.equal(Survey.escapeHtml('"\'&<>'), "&quot;&#39;&amp;&lt;&gt;");
});

test("artifact guard allows only scene-local work paths, not traversal or schemes", () => {
  assert.equal(Survey.artifactUrl("/work/flight_01/survey/evaluation.json", "flight_01"), "/work/flight_01/survey/evaluation.json");
  for (const url of [
    "javascript:alert(1)", "https://example.com/work/flight_01/a.json", "//evil.test/work/a",
    "/work/other/report.json", "/work/flight_01/../secret", "/work/flight_01/%2e%2e/secret",
    "/work/flight_01/%252e%252e/secret", "/work/flight_01/a%2fb", "/work/flight_01/a\\b",
    "/work/flight_01/a?redirect=evil", "/work/flight_01/a#fragment", "/work/flight_01/\na"
  ]) assert.equal(Survey.artifactUrl(url, "flight_01"), null, url);
});

test("numbers distinguish unknown from a real zero and GPS fit from accuracy", () => {
  for (const value of [null, undefined, "0", NaN, Infinity, -1]) {
    assert.equal(Survey.formatMeasurement(value, "m"), "Unknown");
  }
  assert.equal(Survey.formatMeasurement(0, "m"), "0.00 m");
  const html = Survey.renderAlignment({ scale: 2, matched_count: 18, inlier_count: 15, fit_rmse_m: 0.2, accuracy_validated: false });
  assert.match(html, /0\.20 m/);
  assert.match(html, /GPS fit/);
  assert.match(html, /not independent accuracy/i);
  assert.doesNotMatch(html, /accuracy passed/i);
});

test("scene changes discard old evidence and late successes or failures", () => {
  let state = Survey.initialState("flight_01");
  state = Survey.reduceState(state, { type: "begin", id: 1, operation: "load" });
  state = Survey.reduceState(state, { type: "select", scene: "flight_02" });
  assert.equal(state.data, null);
  assert.equal(state.busy, false);
  assert.equal(Survey.reduceState(state, { type: "resolve", id: 1, data: response() }), state);
  assert.equal(Survey.reduceState(state, { type: "reject", id: 1, error: "old error" }), state);
});

test("latest request wins and response for a different scene is rejected", () => {
  let state = Survey.initialState("flight_01");
  state = Survey.reduceState(state, { type: "begin", id: 1, operation: "load" });
  state = Survey.reduceState(state, { type: "begin", id: 2, operation: "load" });
  assert.equal(Survey.reduceState(state, { type: "resolve", id: 1, data: response() }), state);
  state = Survey.reduceState(state, { type: "resolve", id: 2, data: response({ scene: "other" }) });
  assert.equal(state.data, null);
  assert.match(state.error, /scene/i);
  assert.equal(state.busy, false);
});

test("unsupported API schema fails closed and never enables mutation actions", () => {
  let state = Survey.reduceState(Survey.initialState("flight_01"), { type: "begin", id: 1, operation: "load" });
  state = Survey.reduceState(state, { type: "resolve", id: 1, data: response({ schema_version: 2 }) });
  assert.match(state.error, /schema/i);
  assert.equal(Survey.actionAvailability(state).prepare, false);
});

test("busy, empty and offline states disable mutations; prepared scene enables alignment", () => {
  assert.equal(Survey.actionAvailability(Survey.initialState()).save, false);
  let state = Survey.reduceState(Survey.initialState("flight_01"), { type: "begin", id: 1, operation: "load" });
  assert.ok(Object.values(Survey.actionAvailability(state)).every(v => v === false));
  state = Survey.reduceState(state, { type: "resolve", id: 1, data: response({ status: "prepared" }) });
  assert.equal(Survey.actionAvailability(state).align, true);
  assert.equal(Survey.actionAvailability(state).evaluate, true);
  state = Survey.reduceState(state, { type: "begin", id: 2, operation: "align" });
  state = Survey.reduceState(state, { type: "reject", id: 2, error: "API offline" });
  assert.equal(state.busy, false);
  assert.equal(state.data, null);
  assert.equal(Survey.actionAvailability(state).align, false);
});

test("JSON errors and non-JSON failures produce useful operator errors", async () => {
  await assert.rejects(Survey.readResponse(new Response(JSON.stringify({ error: "Inputs already exist" }), { status: 409 })), /409.*Inputs already exist/);
  await assert.rejects(Survey.readResponse(new Response("<html>offline</html>", { status: 503 })), /503/);
  await assert.rejects(Survey.readResponse(new Response("not json", { status: 200 })), /JSON/i);
  await assert.rejects(Survey.readResponse(new Response(JSON.stringify({ error: "Invalid telemetry" }), { status: 200 })), /Invalid telemetry/);
  assert.deepEqual(await Survey.readResponse(new Response(JSON.stringify(response()))), response());
});

test("input payload enforces scene safety, JSON object, UTF-8 size and non-empty CSV", () => {
  const metadata = JSON.stringify({ schema_version: 1 });
  assert.equal(Survey.inputPayload("flight_01", "header\n1", metadata).scene, "flight_01");
  assert.throws(() => Survey.inputPayload("../bad", "header", metadata), /scene/i);
  assert.throws(() => Survey.inputPayload("flight_01", "", metadata), /CSV/i);
  assert.throws(() => Survey.inputPayload("flight_01", "header", "[]"), /object/i);
  assert.throws(() => Survey.inputPayload("flight_01", "header", "{broken"), /JSON/i);
  assert.throws(() => Survey.inputPayload("flight_01", "é".repeat(1100000), metadata), /2 MB/i);
});

test("templates have an empty CSV and exact declared metadata, not fabricated samples", () => {
  assert.equal(Survey.CSV_TEMPLATE.trim().split(/\r?\n/).length, 1);
  assert.deepEqual(JSON.parse(Survey.METADATA_TEMPLATE), {
    schema_version: 1, time_reference: "video", time_offset_s: 0,
    altitude_datum: "ellipsoidal", position_reference: "camera_center",
    single_pass: true, video_duration_s: 600
  });
});

test("command text remains inert and preserves argument boundaries", () => {
  assert.equal(Survey.commandText({ argv: ["python", "pipeline.py", "run", "a b", "$(evil)"] }), 'python pipeline.py run \'a b\' \'$(evil)\'');
  assert.equal(Survey.commandText({ argv: "python bad" }), "Command unavailable");
  assert.equal(Survey.commandText({ argv: ["python", null] }), "Command unavailable");
});

test("server speed gates and reasons are visible without a fabricated accuracy pass", () => {
  const html = Survey.renderCriteria({ criteria: [
    { id: "accuracy", weight: 30, status: "measured", reason: "No official statistic specified", metrics: { rmse_3d_m: 0.25, p95_3d_m: 0.6 } },
    { id: "speed", weight: 20, status: "exceeds_target", reason: "Qualified full run is too slow", metrics: { elapsed_s: 901, video_duration_s: 600 } }
  ] });
  assert.equal(Survey.statusInfo("meets_target").label, "Meets target");
  assert.equal(Survey.statusInfo("exceeds_target").tone, "bad");
  assert.match(html, /Exceeds target/);
  assert.match(html, /0\.25 m/);
  assert.match(html, /No official statistic specified<\/p>/);
  assert.doesNotMatch(html, /Accuracy passed/);
});

test("telemetry template uses all six required server columns and bounded scene names", () => {
  assert.equal(Survey.CSV_TEMPLATE, "t_sec,latitude_deg,longitude_deg,altitude_m,horizontal_std_m,vertical_std_m\n");
  assert.equal(Survey.validScene("flight.01"), false);
  assert.equal(Survey.validScene("a".repeat(65)), false);
  assert.equal(Survey.validScene("flight-01_A"), true);
});

test("inactive View always resolves to blank, including after a model was open", () => {
  const model = "/viewer/pc.html?asset=/work/flight_01/viewer_assets";
  assert.equal(Survey.viewFrameSource("survey", model), "about:blank");
  assert.equal(Survey.viewFrameSource("run", model), "about:blank");
  assert.equal(Survey.viewFrameSource("capture", model), "about:blank");
  assert.equal(Survey.viewFrameSource("view", null), "about:blank");
  assert.equal(Survey.viewFrameSource("view", model), model);
});

/* ---- console rendering: the instrument strip, ledger states, evidence ---- */

test("instrument strip keeps GPS fit and held-out accuracy as two separate readouts", () => {
  const html = Survey.renderInstrument({
    alignment: { fit_rmse_m: 0.16, scale: 1, matched_count: 4, inlier_count: 4 },
    evaluation: { criteria: [{ id: "accuracy", status: "measured", metrics: { rmse_3d_m: 0.42, p95_3d_m: 0.8 } }] }
  }, "flight_01");
  assert.match(html, /GPS fit RMSE/);
  assert.match(html, /Held-out 3D accuracy/);
  assert.match(html, /0\.16 m/);
  assert.match(html, /0\.42 m/);
  assert.match(html, /not independent accuracy/i);
  assert.doesNotMatch(html, /accuracy passed|overall|composite|score/i);
});

test("instrument strip reads the CRS, vertical datum and checkpoint count the server sends", () => {
  const html = Survey.renderInstrument({
    evaluation: { crs: "EPSG:4979 ENU, origin 12.34N 100.98E", vertical_datum: "ellipsoidal",
      counts: { checkpoints: 16 }, criteria: [{ id: "accuracy", status: "measured", metrics: { rmse_3d_m: 0.42 } }] }
  }, "flight_01");
  assert.match(html, /EPSG:4979 ENU, origin 12\.34N 100\.98E/);
  assert.match(html, /ellipsoidal/);
  assert.match(html, /Checkpoints<\/dt><dd>16<\/dd>/);
  const sparse = Survey.renderInstrument({ alignment: { coordinate_frame: "ENU @ scene origin" } }, "flight_01");
  assert.match(sparse, /ENU @ scene origin/);
  assert.match(sparse, /Vertical datum<\/span><span class="survey-ro-v">not available/);
});

test("absent server fields read not available instead of a placeholder number", () => {  const html = Survey.renderInstrument({}, "");
  for (const label of ["CRS / horizontal frame", "Vertical datum", "Position reference", "Run total"]) {
    assert.match(html, new RegExp(label));
  }
  assert.match(html, /not available/);
  assert.equal(Survey.readNumber(undefined, "m"), "not available");
  assert.equal(Survey.readNumber(null, "m"), "not available");
  assert.equal(Survey.readNumber("0.4", "m"), "not available");
  assert.equal(Survey.readNumber(0, "m"), "0.00 m");
  assert.equal(Survey.readNumber(-1, "m"), "not available");
  assert.equal(Survey.formatMeasurement(undefined, "m"), "Unknown");
});

test("stage timings are never invented when the server publishes no breakdown", () => {
  assert.match(Survey.renderStageTimings({}), /not available/i);
  assert.match(Survey.renderStageTimings({ latest_run: { secs: 40 } }), /not available/i);
  assert.match(Survey.renderStageTimings({ latest_run: { secs: 40, error: "boom" } }), /not available/i);
  assert.doesNotMatch(Survey.renderStageTimings({ latest_run: { secs: 40 } }), /<table|width:/);
  const html = Survey.renderStageTimings({ latest_run: { steps: [{ name: "colmap", status: "done", secs: 300.25 }, { name: "poses" }] } });
  assert.match(html, /colmap/);
  assert.match(html, /300\.3/);
  assert.doesNotMatch(html, /poses/, "an untimed stage must not gain a number");
});

test("workflow stages come only from the server status and never imply a pass", () => {
  const aligned = Survey.renderStages("aligned");
  assert.equal((aligned.match(/survey-stage"/g) || []).length, 5);
  assert.match(aligned, /data-state="current"/);
  assert.match(aligned, /Evidence<\/span>\s*<span class="survey-stage-state">not reached/);
  assert.match(Survey.renderStages(undefined), /not reached/);
  assert.doesNotMatch(Survey.renderStages("evaluated"), /passed|approved/i);
});

test("formats checklist reports only extensions the artifacts list contains", () => {
  const artifacts = [
    { name: "evidence_points.ply", url: "/work/flight_01/survey/evidence_points.ply" },
    { name: "scene.obj", url: "/work/flight_01/export/scene.obj" },
    { name: "evaluation.json", url: "/work/flight_01/survey/evaluation.json" }
  ];
  const report = Survey.formatsReport(artifacts);
  assert.deepEqual(report.map(f => f.label), ["OBJ", "PLY", "LAS", "GeoTIFF", "glTF", "FBX"]);
  assert.equal(report.filter(f => f.files.length).map(f => f.label).join(","), "OBJ,PLY");
  const html = Survey.renderFormats(artifacts);
  assert.equal((html.match(/data-state="absent"/g) || []).length, 4);
  assert.match(html, /not delivered/);
  assert.match(html, /LAS/);
  assert.doesNotMatch(html, /pending|scheduled/i);
  assert.equal(Survey.formatsReport(undefined).filter(f => f.files.length).length, 0);
});

test("measurements section states the empty case and escapes returned rows", () => {
  const empty = Survey.renderMeasurements({});
  assert.match(empty, /No measurements exist yet/);
  assert.doesNotMatch(empty, /<td|0\.0|n\/a/i);
  const rows = Survey.renderMeasurements({
    measurements: [
      { label: "<img src=x>", value: 214.5, unit: "pts/m2", method: "<b>voxel</b>", source: "e.json" },
      { label: "Surface residual", value: "0.31 m", method: "held-out pairs" },
      { label: "Odd metric", value: null }
    ]
  });
  assert.match(rows, /214\.50 pts\/m2/);
  assert.match(rows, /0\.31 m/);
  assert.match(rows, /not available/);
  assert.doesNotMatch(rows, /<img |<b>/);
});

test("ledger separates measured, not evaluated, not demonstrated and outside target", () => {
  const html = Survey.renderCriteria({ criteria: [
    { id: "accuracy", status: "measured", metrics: { rmse_3d_m: 0.16 } },
    { id: "completeness", status: "not_evaluated" },
    { id: "innovation", status: "not_demonstrated" },
    { id: "speed", status: "exceeds_target", metrics: { elapsed_s: 901, video_duration_s: 600 } }
  ] });
  assert.deepEqual([...html.matchAll(/class="survey-criterion" data-state="([a-z-]+)"/g)].map(m => m[1]),
    ["measured", "not-evaluated", "over", "not-demonstrated", "unknown", "unknown"]);
  assert.match(html, /Not demonstrated/);
  assert.match(html, /Exceeds target/);
  assert.match(html, /901\.0 s/);
  assert.equal(Survey.stateCode("invented"), "unknown");
  assert.equal(Survey.stateCode("measured"), "measured");
});

test("new console renderers escape every server string", () => {
  const evil = '<img src=x onerror="alert(1)">';
  const html = Survey.renderInstrument({
    crs: evil, vertical_datum: evil, position_reference: evil,
    alignment: { coordinate_frame: evil, fit_rmse_m: 0.1 },
    latest_run: { id: evil, status: evil, secs: 5, steps: [{ name: evil, status: evil, secs: 1 }] }
  }, "flight_01");
  assert.doesNotMatch(html, /<img /);
  assert.match(html, /&lt;img src=x onerror=&quot;alert\(1\)&quot;&gt;/);
  const timings = Survey.renderStageTimings({ latest_run: { steps: [{ name: evil, status: evil, secs: 1 }] } });
  assert.doesNotMatch(timings, /<img /);
  assert.match(Survey.renderStages("prepared"), /data-state="done"/);
});

test("console never renders an aggregate score or a percentage of total weight", () => {
  const data = { alignment: { fit_rmse_m: 0.16 }, evaluation: { criteria: [{ id: "accuracy", status: "measured", metrics: { rmse_3d_m: 0.16 } }] }, latest_run: { secs: 10 } };
  const all = Survey.renderInstrument(data, "flight_01") + Survey.renderCriteria(data.evaluation) +
    Survey.renderStageTimings(data) + Survey.renderFormats(data.artifacts) + Survey.renderMeasurements(data);
  assert.doesNotMatch(all, /100%|overall|aggregate|total score|weighted score/i);
});

