"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { Suspense, useRef } from "react";
import { Icon, type IconName } from "./studio-icons";

const NAV: { href: string; label: string; icon: IconName; match: (path: string, view: string | null) => boolean }[] = [
  { href: "/", label: "Projects", icon: "grid", match: (path, view) => (path === "/" && view !== "processing") || path.startsWith("/projects") },
  { href: "/applications", label: "Apps", icon: "apps", match: (path) => path.startsWith("/applications") },
  { href: "/challenges", label: "Challenges", icon: "target", match: (path) => path.startsWith("/challenges") },
  { href: "/?view=processing", label: "Activity", icon: "activity", match: (path, view) => path === "/" && view === "processing" },
];

function Rail({ connected, onGuide }: { connected: boolean; onGuide: () => void }) {
  const path = usePathname();
  const view = useSearchParams().get("view");
  return <aside className="studio-sidebar">
    <Link href="/" className="studio-brand" aria-label="Ground Control projects" title="Ground Control"><Icon name="cube" size={20} /></Link>
    <nav aria-label="Studio navigation">
      {NAV.map((item) => <Link key={item.href} href={item.href} className={`nav-item${item.match(path, view) ? " active" : ""}`} aria-current={item.match(path, view) ? "page" : undefined}><Icon name={item.icon} size={19} /><span>{item.label}</span></Link>)}
    </nav>
    <div className="sidebar-bottom">
      <button className="nav-item" onClick={onGuide} title="Capture guide"><Icon name="help" size={19} /><span>Guide</span></button>
      <div className="service-status" aria-label={connected ? "Local reconstruction service connected" : "Local service disconnected"}>
        <span className={`status-dot ${connected ? "online" : "offline"}`} />
        <span className="service-tip">{connected ? "Local service online" : "Service offline"}<small>Your data stays on this machine</small></span>
      </div>
    </div>
  </aside>;
}

export function Studio({ children, project, connected = true }: { children: React.ReactNode; project?: string; connected?: boolean }) {
  const guide = useRef<HTMLDialogElement>(null);
  return <div className={`studio ${project ? "studio-workspace" : ""}`}>
    <Suspense fallback={<aside className="studio-sidebar" />}><Rail connected={connected} onGuide={() => guide.current?.showModal()} /></Suspense>
    <div className="studio-content">{children}</div>
    <dialog ref={guide} className="studio-dialog guide-dialog" onClick={(event) => { if (event.target === guide.current) guide.current?.close(); }}>
      <div className="dialog-heading"><div><span className="eyebrow">Capture guide</span><h2>One pass. Make it count.</h2><p>Drone, phone or handheld footage works. Reconstruction needs camera movement and visible texture — not a camera spinning in place.</p></div><button className="icon-button" aria-label="Close capture guide" onClick={() => guide.current?.close()}><Icon name="close" /></button></div>
      <div className="guide-content"><ol><li><span><strong>Move steadily.</strong>Keep neighbouring frames overlapping; avoid fast turns and digital zoom.</span></li><li><span><strong>Keep detail sharp.</strong>Good light and the original file. Blur and heavy compression throw away geometry.</span></li><li><span><strong>Capture what matters.</strong>Hidden facades and occluded surfaces cannot be measured from a single pass.</span></li><li><span><strong>Bring a reference.</strong>Pose logs, known camera height, telemetry and independent checkpoints each serve a different purpose.</span></li></ol><div className="notice"><Icon name="info" /><p>A convincing model is not proof of survey accuracy. The studio keeps scale, georeferencing and independent accuracy as three separate claims.</p></div></div>
    </dialog>
  </div>;
}

export function Status({ status }: { status: string }) {
  const label: Record<string, string> = { ready: "Ready", processing: "Processing", failed: "Needs attention", uploaded: "Ready to process", empty: "No video yet" };
  return <span className={`project-status status-${status}`}><span />{label[status] ?? status}</span>;
}

export function Offline({ message, retry }: { message: string; retry: () => void }) {
  return <section className="empty-state offline-state"><span className="empty-icon"><Icon name="activity" size={28} /></span><h2>Reconstruction service offline</h2><p>Start the local Python service to open your projects. Nothing has been deleted.</p><code>.venv/Scripts/python.exe pipeline.py ui</code><p className="error-detail">{message}</p><button className="button primary" onClick={retry}><Icon name="reset" size={16} />Retry connection</button></section>;
}
