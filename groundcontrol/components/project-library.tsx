"use client";

import Link from "next/link";
import Image from "next/image";
import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { WORKFLOWS, date, displayName, workspace, type ProjectList } from "@/lib/workspace";
import { Studio, Status, Offline } from "./studio";
import { Icon } from "./studio-icons";
import { ImportDialog } from "./import-dialog";

export function ProjectLibrary() {
  const query = useSearchParams();
  const [data, setData] = useState<ProjectList | null>(null);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [open, setOpen] = useState(false);
  const refresh = useCallback(async () => {
    try { setData(await workspace.list()); setError(""); }
    catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
  }, []);
  useEffect(() => { let active = true; let timer: ReturnType<typeof setTimeout>; const poll = async () => { if (!active) return; await refresh(); if (active) timer = setTimeout(poll, 5000); }; void poll(); return () => { active = false; clearTimeout(timer); }; }, [refresh]);
  const view = query.get("view") ?? "all";
  const activeFilter = ["all", "ready", "processing", "failed"].includes(view) ? view : "all";
  const projects = [...(data?.projects ?? [])].sort((a, b) => b.updated.localeCompare(a.updated));
  const filtered = projects.filter((project) => (activeFilter === "all" || project.status === activeFilter) && `${project.name} ${project.id}`.toLowerCase().includes(search.toLowerCase()));
  const ready = projects.filter((project) => project.viewable).length;
  return <Studio connected={!error}>
    <main className="library-main">
      <div className="page-title"><div><div className="eyebrow">YOUR RECONSTRUCTION WORKSPACE</div><h1>Projects<span aria-hidden="true" className="title-count">{projects.length || "—"}</span></h1><p>A new perspective on every capture.</p></div><button className="button primary" onClick={() => setOpen(true)}><Icon name="plus" size={18} />New reconstruction</button></div>
      <div className="library-summary"><span><span className="status-dot online" /><strong>{ready}</strong> ready to explore</span><span><Icon name="video" size={15} /><strong>{projects.reduce((sum, project) => sum + project.video_count, 0)}</strong> source videos</span><span className="summary-message">Capture → Reconstruct → Explore</span></div>
      <div className="library-toolbar"><div className="filter-tabs" aria-label="Filter projects">{[["all", "All projects"], ["ready", "Ready"], ["processing", "Processing"], ["failed", "Needs attention"]].map(([value, label]) => <Link key={value} href={value === "all" ? "/" : `/?view=${value}`} className={activeFilter === value ? "active" : ""}>{label}</Link>)}</div><label className="search-field"><Icon name="search" size={17} /><input aria-label="Search projects" placeholder="Search projects…" value={search} onChange={(event) => setSearch(event.target.value)} /></label></div>
      {error ? <Offline message={error} retry={() => void refresh()} /> : !data ? <div className="loading-state" role="status"><span className="spinner" />Opening your workspace…</div> : filtered.length === 0 ? <div className="empty-state"><span className="empty-icon"><Icon name={search ? "search" : "cube"} size={32} /></span><h2>{search ? "No matching projects" : activeFilter === "processing" ? "Nothing processing right now" : "Your next perspective starts here"}</h2><p>{search ? "Try a different project name." : "Import a video to reconstruct a scene, or open an existing project."}</p>{!search && <button className="button secondary" onClick={() => setOpen(true)}><Icon name="upload" size={16} />Import a video</button>}</div> : <div className="project-grid">{filtered.map((project) => <Link href={`/projects/${encodeURIComponent(project.id)}`} key={project.id} className="project-card" aria-label={`Open ${displayName(project)}`}>
        <div className="project-preview">{project.thumbnail_url ? <Image unoptimized width={640} height={400} src={project.thumbnail_url} alt={`Source frame from ${displayName(project)}`} loading="lazy" /> : <div className="no-preview"><Icon name="cube" size={42} /><span>{project.video_count ? "Ready for reconstruction" : "Awaiting capture"}</span></div>}<div className="preview-top"><span className="workflow-tag">{WORKFLOWS[project.workflow]?.label ?? "Explore"}</span><span className="preview-arrow"><Icon name="arrow" size={18} /></span></div><span className="preview-label"><Icon name="image" size={12} />{project.thumbnail_url ? "Source frame" : "No preview yet"}</span>{project.viewable && <span className="model-tag"><Icon name="cube" size={13} />3D available</span>}</div>
        <div className="project-card-body"><div className="card-heading"><h2>{displayName(project)}</h2><Icon name="chevron" size={16} /></div><div className="project-meta"><span>{project.registered_count.toLocaleString()} cameras</span><span>{project.scale.status === "metric" ? "Metric scale" : project.scale.status === "estimated" ? "Estimated scale" : "Relative scale"}</span></div><div className="card-footer"><Status status={project.status} /><time>{date(project.updated)}</time></div></div>
      </Link>)}</div>}
      <footer className="library-footer"><Icon name="folder" size={14} /><span>Projects are read from your local captures and reconstruction outputs.</span><span>Ground Control Studio</span></footer>
    </main>
    {open && <ImportDialog open={open} onClose={() => setOpen(false)} />}
  </Studio>;
}
