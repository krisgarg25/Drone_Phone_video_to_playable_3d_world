"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { APPLICATION_PROFILES, OFFICIAL_FORMATS, WORKSPACE_TABS, workspaceHref } from "@/lib/applications";
import { displayName, workspace, type Project } from "@/lib/workspace";
import { Studio } from "./studio";
import { Icon } from "./studio-icons";

const PICK_KEY = "gc.applications.scene";

export function ApplicationsHub() {
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [error, setError] = useState("");
  const [picked, setPicked] = useState("");
  useEffect(() => {
    workspace.list().then((list) => {
      try { setPicked(localStorage.getItem(PICK_KEY) ?? ""); } catch { /* storage unavailable: default scene */ }
      setProjects(list.projects.filter((p) => p.viewable).sort((a, b) => b.updated.localeCompare(a.updated)));
    }).catch((cause) => setError(cause instanceof Error ? cause.message : String(cause)));
  }, []);
  const scene = useMemo(() => projects?.find((p) => p.id === picked) ?? projects?.[0] ?? null, [projects, picked]);
  const choose = (id: string) => { setPicked(id); try { localStorage.setItem(PICK_KEY, id); } catch { /* per-viewer convenience only */ } };
  // A scene built for this application opens its tools; otherwise the scene chosen above.
  const sceneFor = (id: string) => projects?.find((p) => p.application?.id === id) ?? scene;
  const analyses = APPLICATION_PROFILES.reduce((n, app) => n + app.analyses.length, 0);

  return <Studio connected={!error}>
    <main className="library-main">
      <section className="hub-hero">
        <div>
          <div className="eyebrow">SIH 26158 · Potential applications</div>
          <h1>Eight applications.<br /><em>One reconstruction.</em></h1>
          <p style={{ marginTop: 16, maxWidth: "58ch" }}>Every application is a set of tools on the same single-pass model. Pick a scene, then open any application&apos;s workspace directly — the tools, the analyses and the official outputs it is judged on.</p>
        </div>
        <div className="hub-scene">
          <label>Open tools on
            <select value={scene?.id ?? ""} onChange={(event) => choose(event.target.value)} disabled={!projects?.length} aria-label="Scene the tools open on">
              {!projects && <option>Loading scenes…</option>}
              {projects?.length === 0 && <option>No 3D scenes yet</option>}
              {projects?.map((p) => <option key={p.id} value={p.id}>{displayName(p)}</option>)}
            </select>
          </label>
          <p>{error ? "The local service is offline, so tool links are disabled." : "Where a scene was built for an application, its tools open on that scene instead."}</p>
        </div>
      </section>

      <div className="library-summary" style={{ marginTop: 0 }}>
        <div className="stat"><span><Icon name="apps" size={14} />Applications</span><strong>8</strong></div>
        <div className="stat"><span><Icon name="grid" size={14} />Workspaces</span><strong>{Object.keys(WORKSPACE_TABS).length}</strong></div>
        <div className="stat"><span><Icon name="activity" size={14} />Analyses</span><strong>{analyses}</strong></div>
        <div className="stat"><span><Icon name="file" size={14} />Official formats</span><strong>{OFFICIAL_FORMATS.length}</strong></div>
      </div>

      <div className="app-grid">
        {APPLICATION_PROFILES.map((app, index) => {
          const target = sceneFor(app.id);
          return <article key={app.id} className="app-card" style={{ ["--app-hue" as string]: app.hue, animationDelay: `${index * 0.04}s` }}>
            <header className="app-head">
              <span className="app-icon"><Icon name={app.icon} size={22} /></span>
              <div style={{ flex: 1, minWidth: 0 }}><h2>{app.label}</h2><p>{app.summary}</p></div>
              <span className="app-index">{String(index + 1).padStart(2, "0")}</span>
            </header>
            <div className="app-block"><span>Open in workspace</span>
              <div className="app-tabs">{app.tabs.map((tab) => <Link key={tab} className={`app-tab${target ? "" : " disabled"}`} href={target ? workspaceHref(target.id, tab) : "#"} aria-disabled={!target}><Icon name={WORKSPACE_TABS[tab].icon} size={14} />{WORKSPACE_TABS[tab].label}</Link>)}</div>
            </div>
            <div className="app-block"><span>Analyses</span><div className="app-analyses">{app.analyses.map((a) => <span key={a}>{a}</span>)}</div></div>
            <div className="app-block"><span>Official outputs</span><div className="app-products">{app.products.map((p) => <span key={p}>{p}</span>)}</div></div>
            <div className="app-block"><span>Acceptance checks</span><ul className="app-checks">{app.checks.map((c) => <li key={c}><Icon name="check" size={13} />{c}</li>)}</ul></div>
            <footer className="app-foot"><span>Best captured as {app.patterns.join(" or ")} flight</span><span>{target ? <>Opens on <b style={{ color: "hsl(var(--st-ink))", fontWeight: 500 }}>{displayName(target)}</b></> : "No scene available"}</span></footer>
          </article>;
        })}
      </div>
    </main>
  </Studio>;
}
