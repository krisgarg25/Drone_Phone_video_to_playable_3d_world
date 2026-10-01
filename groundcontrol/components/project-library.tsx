"use client";

import Link from "next/link";
import Image from "next/image";
import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { WORKFLOWS, date, displayName, workspace, type ProjectList } from "@/lib/workspace";
import { Studio, Status, Offline } from "./studio";
import { Icon } from "./studio-icons";
import { ImportDialog } from "./import-dialog";

const FILTERS = [["all", "All"], ["ready", "Ready"], ["processing", "Processing"], ["failed", "Needs attention"]] as const;

export function ProjectLibrary() {
  const query = useSearchParams();
  const [data, setData] = useState<ProjectList | null>(null);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [open, setOpen] = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);
  const refresh = useCallback(async () => {
    try { setData(await workspace.list()); setError(""); }
    catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
  }, []);
  useEffect(() => { let active = true; let timer: ReturnType<typeof setTimeout>; const poll = async () => { if (!active) return; await refresh(); if (active) timer = setTimeout(poll, 5000); }; void poll(); return () => { active = false; clearTimeout(timer); }; }, [refresh]);
  useEffect(() => {
    const key = (event: KeyboardEvent) => { if (event.key === "/" && document.activeElement?.tagName !== "INPUT") { event.preventDefault(); searchRef.current?.focus(); } };
    window.addEventListener("keydown", key); return () => window.removeEventListener("keydown", key);
  }, []);
  const view = query.get("view") ?? "all";
  const activeFilter = ["all", "ready", "processing", "failed"].includes(view) ? view : "all";
  const projects = [...(data?.projects ?? [])].sort((a, b) => b.updated.localeCompare(a.updated));
  const filtered = projects.filter((project) => (activeFilter === "all" || project.status === activeFilter) && `${project.name} ${project.id}`.toLowerCase().includes(search.toLowerCase()));
  const count = (status: string) => status === "all" ? projects.length : projects.filter((project) => project.status === status).length;
  const ready = projects.filter((project) => project.viewable).length;
  const cameras = projects.reduce((sum, project) => sum + project.registered_count, 0);
  const metric = projects.filter((project) => project.scale.status === "metric").length;
  const videos = projects.reduce((sum, project) => sum + project.video_count, 0);
  const n = (value: number) => data ? value.toLocaleString() : "—";
  return <Studio connected={!error}>
    <main className="library-main">
      <div className="page-title">
        <div><div className="eyebrow">Reconstruction studio</div><h1>Projects<span aria-hidden="true" className="title-count">{data ? projects.length : ""}</span></h1><p>Single-pass drone video in; a measurable, georeferenced 3D workspace out. Everything runs on this machine.</p></div>
        <div className="page-actions"><Link className="button secondary" href="/applications"><Icon name="apps" size={16} />Applications</Link><button className="button primary" onClick={() => setOpen(true)}><Icon name="plus" size={16} />New reconstruction</button></div>
      </div>
      <div className="library-summary">
        <div className="stat live"><span><Icon name="cube" size={14} />3D workspaces ready</span><strong>{n(ready)}<small>/ {n(projects.length)}</small></strong></div>
        <div className="stat"><span><Icon name="camera" size={14} />Cameras recovered</span><strong>{n(cameras)}</strong></div>
        <div className="stat"><span><Icon name="ruler" size={14} />Metric-scale scenes</span><strong>{n(metric)}</strong></div>
        <div className="stat"><span><Icon name="video" size={14} />Source videos</span><strong>{n(videos)}</strong></div>
      </div>
      <div className="library-toolbar">
        <div className="segmented filter-tabs" aria-label="Filter projects">{FILTERS.map(([value, label]) => <Link key={value} href={value === "all" ? "/" : `/?view=${value}`} className={activeFilter === value ? "active" : ""}>{label}<span className="seg-count">{data ? count(value) : ""}</span></Link>)}</div>
        <label className="search-field"><Icon name="search" size={16} /><input ref={searchRef} aria-label="Search projects" placeholder="Search projects" value={search} onChange={(event) => setSearch(event.target.value)} /><kbd>/</kbd></label>
      </div>
      {error ? <Offline message={error} retry={() => void refresh()} />
        : !data ? <div className="project-grid" role="status" aria-label="Loading projects">{[0, 1, 2, 3, 4, 5].map((i) => <div key={i} className="skeleton" />)}</div>
        : filtered.length === 0 ? <div className="empty-state"><span className="empty-icon"><Icon name={search ? "search" : "cube"} size={28} /></span><h2>{search ? "No matching projects" : activeFilter === "processing" ? "Nothing processing right now" : "Your next perspective starts here"}</h2><p>{search ? "Try a different project name." : activeFilter === "processing" ? "Reconstructions you start appear here with live progress." : "Import a single-pass video to reconstruct a scene."}</p>{!search && <button className="button primary" onClick={() => setOpen(true)}><Icon name="upload" size={16} />Import a video</button>}</div>
        : <div className="project-grid">{filtered.map((project) => <Link href={`/projects/${encodeURIComponent(project.id)}`} key={project.id} className="project-card" aria-label={`Open ${displayName(project)}`}>
          <div className="project-preview">
            {project.thumbnail_url ? <Image unoptimized width={640} height={400} src={project.thumbnail_url} alt={`Source frame from ${displayName(project)}`} loading="lazy" /> : <div className="no-preview"><Icon name="cube" size={34} /><span>{project.video_count ? "Ready for reconstruction" : "Awaiting capture"}</span></div>}
            <div className="preview-top"><span className="workflow-tag">{project.scenario?.label || WORKFLOWS[project.workflow]?.label || "Explore"}</span><span className="preview-arrow"><Icon name="arrowUpRight" size={16} /></span></div>
            {project.viewable && <span className="model-tag"><Icon name="cube" size={13} />3D ready</span>}
          </div>
          <div className="project-card-body">
            <div className="card-heading"><h2>{displayName(project)}</h2><Icon name="chevron" size={16} /></div>
            <div className="project-meta">
              <span><b>{project.registered_count.toLocaleString()}</b> cameras</span>
              <span>{project.scale.status === "metric" ? "Metric" : project.scale.status === "estimated" ? "Est. scale" : "Relative"}</span>
              {project.quality ? <span className={`quality-badge tone-${project.quality.status}`} title={project.quality.hard_failures.length ? `Blocked by: ${project.quality.hard_failures.join(", ")}` : project.quality.warnings.join(", ") || "All world-gate checks passed"}>{project.quality.status === "failed" ? `Gate blocked · ${project.quality.hard_failures.length}` : project.quality.status === "warnings" ? `${project.quality.warnings.length} limit${project.quality.warnings.length === 1 ? "" : "s"}` : "Gate passed"}</span> : null}
            </div>
            <div className="card-footer"><Status status={project.status} /><time>{date(project.updated)}</time></div>
          </div>
        </Link>)}</div>}
      <footer className="library-footer"><Icon name="folder" size={14} /><span>Projects are read from your local captures and reconstruction outputs.</span><span>SIH 26158 · Ground Control</span></footer>
    </main>
    {open && <ImportDialog open={open} onClose={() => setOpen(false)} />}
  </Studio>;
}
