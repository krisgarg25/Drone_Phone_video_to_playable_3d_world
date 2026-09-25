"use client";

import Link from "next/link";
import { useRef } from "react";
import { Icon } from "./studio-icons";

export function Studio({ children, project, connected = true }: { children: React.ReactNode; project?: string; connected?: boolean }) {
  const guide = useRef<HTMLDialogElement>(null);
  return <div className={`studio ${project ? "studio-workspace" : ""}`}>
    <aside className="studio-sidebar">
      <Link href="/" className="studio-brand" aria-label="Ground Control projects"><span className="brand-symbol"><Icon name="cube" size={24} /></span><span>ground<span className="brand-light">control</span><small>RECONSTRUCTION STUDIO</small></span></Link>
      <span className="nav-caption">WORKSPACE</span>
      <nav aria-label="Workspace navigation">
        <Link href="/" className="nav-item" title="Projects"><Icon name="grid" /><span>Projects</span></Link>
        <Link href="/?view=processing" className="nav-item" title="Processing"><Icon name="activity" /><span>Processing</span></Link>
      </nav>
      <div className="sidebar-note"><Icon name="cube" size={28} /><strong>One capture.<br />A new perspective.</strong><p>Video to a world you can explore.</p></div>
      <div className="sidebar-bottom">
        <button className="nav-item" onClick={() => guide.current?.showModal()} title="Capture guide"><Icon name="help" /><span>Capture guide</span></button>
        <div className="service-status" title={connected ? "Local reconstruction service connected" : "Local service disconnected"}><span className={`status-dot ${connected ? "online" : "offline"}`} /><span>{connected ? "Local workspace" : "Service offline"}<small>Your data stays on this machine</small></span></div>
      </div>
    </aside>
    <div className="studio-content">
      <header className="studio-topbar"><div className="breadcrumb"><Link href="/">Workspace</Link><Icon name="chevron" size={13} /><span>{project || "Projects"}</span></div><span className="local-tag"><Icon name="link" size={13} /> Local processing</span></header>
      {children}
    </div>
    <dialog ref={guide} className="studio-dialog guide-dialog">
      <div className="dialog-heading"><div><span className="eyebrow">CAPTURE GUIDE</span><h2>Give the scene another angle.</h2></div><button className="icon-button" aria-label="Close capture guide" onClick={() => guide.current?.close()}><Icon name="close" /></button></div>
      <div className="guide-content"><p>Drone, phone, or handheld footage can work. Reconstruction needs camera movement and visible features—not just a rotating camera.</p><ol><li><strong>Move steadily.</strong> Keep neighbouring frames overlapping and avoid fast turns or digital zoom.</li><li><strong>Keep detail sharp.</strong> Use good light and the original video. Blur and heavy compression discard geometry.</li><li><strong>Capture what matters.</strong> Hidden facades and occluded surfaces cannot be measured from a single pass.</li><li><strong>Bring a reference.</strong> Optional pose logs, known camera height, telemetry, and independent checkpoints serve different purposes.</li></ol><div className="notice"><Icon name="info" /><p>A visually convincing model is not proof of survey accuracy. The workspace keeps scale, georeferencing, and independent accuracy separate.</p></div></div>
    </dialog>
  </div>;
}

export function Status({ status }: { status: string }) {
  const label: Record<string, string> = { ready: "Ready to explore", processing: "Processing", failed: "Needs attention", uploaded: "Ready to process", empty: "No video yet" };
  return <span className={`project-status status-${status}`}><span />{label[status] ?? status}</span>;
}

export function Offline({ message, retry }: { message: string; retry: () => void }) {
  return <section className="empty-state offline-state"><span className="empty-icon"><Icon name="activity" size={32} /></span><h2>Reconstruction service offline</h2><p>Start the local Python service to open your projects. Nothing has been deleted.</p><code>.venv/Scripts/python.exe pipeline.py ui</code><p className="error-detail">{message}</p><button className="button primary" onClick={retry}><Icon name="reset" size={16} />Retry connection</button></section>;
}
