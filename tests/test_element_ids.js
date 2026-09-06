#!/usr/bin/env node
/*
 * test_element_ids.js
 *
 * Every element a page looks up by id must exist, and every id in the markup
 * must be used by something. The failure mode this guards against is the one
 * that already cost a bug: a HUD element was deleted for layout reasons while
 * its writer survived, so the AR loop threw the first time it degraded.
 *
 * Checks on every page in PAGES (capture, probe, pipeline dashboard):
 *   1. each getElementById("x") literal in the page script has a matching id=""
 *   2. each inline on* handler attribute names a function the page defines
 *   3. each id in the markup is referenced by the script, the CSS, or another
 *      attribute -- unreferenced ids are dead markup
 *   4. each script src points at a file that exists
 *
 * Run:  node tests/test_element_ids.js
 */
"use strict";
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const PAGES = ["viewer/capture.html", "viewer/xr_probe.html", "viewer/pipeline_gui.html"];

let pass = 0, fail = 0;
function check(cond, label, detail) {
  if (cond) { pass++; console.log("pass  " + label); }
  else { fail++; console.log("FAIL  " + label + (detail ? "  -> " + detail : "")); }
}

for (const rel of PAGES) {
  const file = path.join(ROOT, rel);
  console.log("\n--- " + rel + " ---");
  const html = fs.readFileSync(file, "utf8");

  // Script bodies: inline <script> blocks plus the external files they load.
  const inline = [...html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)]
    .map(x => x[1]).join("\n");
  const srcs = [...html.matchAll(/<script[^>]*\bsrc="\/?viewer\/([^"]+)"/g)].map(x => x[1]);
  for (const s of srcs) {
    check(fs.existsSync(path.join(ROOT, "viewer", s)), "viewer/" + s + " exists on disk");
  }
  const libs = srcs.map(s => fs.readFileSync(path.join(ROOT, "viewer", s), "utf8")).join("\n");
  const js = inline + "\n" + libs;

  const ids = new Set([...html.matchAll(/\sid="([^"]+)"/g)].map(x => x[1]));
  const styleText = [...html.matchAll(/<style[^>]*>([\s\S]*?)<\/style>/g)].map(x => x[1]).join("\n");

  // 1. lookups must resolve
  const looked = [...new Set([...js.matchAll(/getElementById\(\s*["']([^"']+)["']\s*\)/g)].map(x => x[1]))];
  const missing = looked.filter(id => !ids.has(id));
  check(missing.length === 0,
        "all " + looked.length + " getElementById lookups resolve to markup",
        missing.map(id => "#" + id).join(" "));

  // 2. inline handlers must name real functions
  const handlers = [...html.matchAll(/\son[a-z]+="([A-Za-z_$][\w$]*)\(/g)].map(x => x[1]);
  const defined = new Set([...js.matchAll(/function\s+([A-Za-z_$][\w$]*)\s*\(/g)].map(x => x[1]));
  const globalFns = new Set(["parseInt", "parseFloat", "Number", "String", "Math", "JSON", "alert"]);
  const ghost = [...new Set(handlers)].filter(fn => !defined.has(fn) && !globalFns.has(fn));
  check(ghost.length === 0,
        "all " + new Set(handlers).size + " inline on* handlers are defined functions",
        ghost.join(" "));

  // 3. no dead markup
  //    A page may build its lookups ("lane-" + name), so a quoted prefix that is
  //    concatenated at runtime counts as a reference to every id it completes.
  const concat = [...new Set([...js.matchAll(/["'`]([A-Za-z][\w]*-)["'`]\s*\+/g)].map(x => x[1]))];
  const dead = [...ids].filter(id => {
    if (new RegExp('["\'`](#|\\.)?' + id.replace(/[-]/g, "\\$&") + '["\'`]').test(js)) return false;
    if (styleText.includes("#" + id)) return false;
    if (new RegExp('\\sid="' + id + '"[^>]*\\son[a-z]+=').test(html)) return false;
    if (new RegExp('href="#' + id + '"').test(html)) return false;
    if (new RegExp('for="' + id + '"').test(html)) return false;
    if (concat.some(p => id.startsWith(p) && id.length > p.length)) return false;
    return true;
  });
  check(dead.length === 0, "no unreferenced ids in markup (" + ids.size + " present)",
        dead.map(id => "#" + id).join(" "));

  // 4. data-tab / data-body pairs must match, since switchTab keys off both.
  //    Markup only: the page builds a data-tab selector out of concatenated
  //    strings at runtime, and that is not a tab declaration.
  const markup = html.replace(/<script[\s\S]*?<\/script>/g, "");
  const tabs = [...new Set([...markup.matchAll(/data-tab="([^"]+)"/g)].map(x => x[1]))];
  const bodies = [...new Set([...markup.matchAll(/data-body="([^"]+)"/g)].map(x => x[1]))];
  const orphanTabs = tabs.filter(t => !bodies.includes(t));
  const orphanBodies = bodies.filter(b => !tabs.includes(b));
  check(orphanTabs.length === 0 && orphanBodies.length === 0,
        "every tab has a body and every body has a tab (" + tabs.join("/") + ")",
        "tabs without body: " + orphanTabs + " | bodies without tab: " + orphanBodies);
}

console.log("");
console.log(fail === 0 ? "ALL " + pass + " REFERENCE CHECKS PASSED"
                       : pass + " passed, " + fail + " FAILED");
process.exit(fail === 0 ? 0 : 1);
