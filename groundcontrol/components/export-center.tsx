"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { zipSync } from "fflate";
import { GROUPS, bytes, cleanName, exportItems, type ExportGroup, type ExportItem } from "@/lib/exports";
import { workspace, type ProjectDetail } from "@/lib/workspace";
import { Studio, Offline } from "./studio";
import { Icon } from "./studio-icons";
import "./export-center.css";

type Step = "waiting" | "working" | "done" | "failed";

function save(name: string, blob: Blob) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 8000);
}

export function ExportCenter({ scene }: { scene: string }) {
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [items, setItems] = useState<ExportItem[] | null>(null);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [names, setNames] = useState<Record<string, string>>({});
  const [formats, setFormats] = useState<Record<string, string>>({});
  const [group, setGroup] = useState<ExportGroup | "all">("all");
  const [search, setSearch] = useState("");
  const [prefix, setPrefix] = useState("");
  const [mode, setMode] = useState<"zip" | "files">("zip");
  const [zipName, setZipName] = useState("");
  const [steps, setSteps] = useState<Record<string, { step: Step; note?: string }>>({});
  const [running, setRunning] = useState(false);
  const [finished, setFinished] = useState(false);
  const dialog = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    let live = true;
    workspace.project(scene).then(async (detail) => {
      const list = await exportItems(detail);
      if (!live) return;
      setProject(detail); setItems(list); setZipName(cleanName(`${detail.id}_export`));
      // The three things people usually want are ticked to start with.
      setSelected(new Set(list.filter((item) => /splat\.ply|textured\.obj|made:measurements/.test(item.id)).map((item) => item.id)));
    }).catch((cause) => { if (live) setError(cause instanceof Error ? cause.message : String(cause)); });
    return () => { live = false; };
  }, [scene]);

  const formatOf = (item: ExportItem) => item.formats.find((f) => f.id === formats[item.id]) ?? item.formats[0];
  const nameOf = (item: ExportItem) => names[item.id] ?? item.defaultName;
  const visible = useMemo(() => (items ?? []).filter((item) => (group === "all" || item.group === group)
    && `${item.title} ${item.detail} ${item.defaultName}`.toLowerCase().includes(search.toLowerCase())), [items, group, search]);
  const chosen = (items ?? []).filter((item) => selected.has(item.id));
  const knownBytes = chosen.reduce((sum, item) => sum + (item.bytes ?? 0), 0);
  const unknown = chosen.some((item) => item.bytes == null);
  const allVisible = visible.length > 0 && visible.every((item) => selected.has(item.id));

  const toggle = (id: string) => setSelected((previous) => { const next = new Set(previous); if (next.has(id)) next.delete(id); else next.add(id); return next; });
  const toggleVisible = () => setSelected((previous) => { const next = new Set(previous); for (const item of visible) { if (allVisible) next.delete(item.id); else next.add(item.id); } return next; });
  const applyPrefix = () => {
    const clean = cleanName(prefix);
    if (!clean) return;
    setNames((previous) => { const next = { ...previous }; for (const item of chosen) next[item.id] = cleanName(`${clean}_${nameOf(item).replace(new RegExp(`^${project?.id}_`), "")}`); return next; });
  };
  const finalNames = () => {
    const used = new Set<string>();
    return chosen.map((item) => {
      const f = formatOf(item);
      let name = `${cleanName(nameOf(item))}.${f.ext}`;
      for (let n = 2; used.has(name.toLowerCase()); n++) name = `${cleanName(nameOf(item))}-${n}.${f.ext}`;
      used.add(name.toLowerCase());
      return { item, format: f, name };
    });
  };

  function review() { setSteps({}); setFinished(false); dialog.current?.showModal(); }
  async function download() {
    const plan = finalNames();
    setRunning(true); setFinished(false);
    setSteps(Object.fromEntries(plan.map(({ item }) => [item.id, { step: "waiting" as Step }])));
    const zipped: Record<string, Uint8Array> = {};
    for (const { item, format, name } of plan) {
      setSteps((previous) => ({ ...previous, [item.id]: { step: "working" } }));
      try {
        const blob = await item.build(format.id);
        if (mode === "zip") zipped[name] = new Uint8Array(await blob.arrayBuffer());
        else { save(name, blob); await new Promise((resolve) => setTimeout(resolve, 350)); }
        setSteps((previous) => ({ ...previous, [item.id]: { step: "done", note: bytes(blob.size) } }));
      } catch (cause) {
        setSteps((previous) => ({ ...previous, [item.id]: { step: "failed", note: cause instanceof Error ? cause.message : String(cause) } }));
      }
    }
    if (mode === "zip" && Object.keys(zipped).length) save(`${cleanName(zipName || `${scene}_export`)}.zip`, new Blob([zipSync(zipped, { level: 0 }) as BlobPart], { type: "application/zip" }));
    setRunning(false); setFinished(true);
  }

  if (error) return <Studio connected={false}><main className="library-main"><Offline message={error} retry={() => location.reload()} /></main></Studio>;
  const doneCount = Object.values(steps).filter((s) => s.step === "done").length;
  const failed = Object.values(steps).filter((s) => s.step === "failed").length;

  return <Studio>
    <main className="export-page">
      <header className="export-head">
        <Link className="workspace-back" href={`/projects/${encodeURIComponent(scene)}`} aria-label="Back to the 3D workspace"><Icon name="back" size={16} /></Link>
        <div>
          <span className="eyebrow">{project?.name ?? scene}</span>
          <h1>Export</h1>
          <p>Pick what to take with you, rename it, choose a format, and download it as one ZIP or separate files.</p>
        </div>
        {project && project.artifacts.length > 0 && <a className="button secondary" href={`/api/backend/api/workspace/bundle?scene=${encodeURIComponent(scene)}`} download={`${scene}-bundle.zip`} data-tip="Every file plus a README of caveats and checksums"><Icon name="folder" size={16} />Whole project as .zip</a>}
      </header>

      <div className="export-layout">
        <nav className="export-groups" aria-label="Export categories">
          <button className={group === "all" ? "active" : ""} onClick={() => setGroup("all")}><Icon name="apps" size={17} /><span>Everything</span><em>{items?.length ?? "…"}</em></button>
          {GROUPS.map((g) => { const count = (items ?? []).filter((item) => item.group === g.id).length; const picked = chosen.filter((item) => item.group === g.id).length;
            return <button key={g.id} className={group === g.id ? "active" : ""} disabled={!count} onClick={() => setGroup(g.id)}><Icon name={g.icon} size={17} /><span>{g.label}<small>{g.what}</small></span><em>{picked ? <b>{picked}/</b> : null}{count}</em></button>; })}
        </nav>

        <section className="export-list" aria-label="Files you can export">
          <div className="export-tools">
            <label className="export-check"><input type="checkbox" checked={allVisible} onChange={toggleVisible} disabled={!visible.length} /><span>{allVisible ? "Clear these" : "Select all shown"}</span></label>
            <label className="search-field"><Icon name="search" size={16} /><input placeholder="Search exports" value={search} onChange={(event) => setSearch(event.target.value)} aria-label="Search exports" /></label>
            <div className="export-prefix" data-tip="Adds this to the start of every selected file name"><input placeholder="Name prefix, e.g. site-A_2026" value={prefix} onChange={(event) => setPrefix(event.target.value)} aria-label="Prefix for selected file names" /><button className="button secondary small" disabled={!prefix.trim() || !chosen.length} onClick={applyPrefix}>Apply to selected</button></div>
          </div>

          {!items ? <div className="export-rows">{[0, 1, 2, 3, 4].map((i) => <div key={i} className="export-skeleton" />)}</div>
            : !visible.length ? <div className="tool-empty"><Icon name="search" size={26} /><p>Nothing matches.</p></div>
            : GROUPS.filter((g) => group === "all" || g.id === group).map((g) => { const rows = visible.filter((item) => item.group === g.id); return rows.length ? <div key={g.id} className="export-section">
              <h2>{g.label}<span>{rows.length}</span></h2>
              <ul className="export-rows">{rows.map((item) => { const on = selected.has(item.id); const f = formatOf(item);
                return <li key={item.id} className={`export-row${on ? " is-on" : ""}`}>
                  <label className="export-pick"><input type="checkbox" checked={on} onChange={() => toggle(item.id)} aria-label={`Include ${item.title}`} /></label>
                  <span className="export-icon"><Icon name={item.icon} size={18} /></span>
                  <div className="export-what"><strong>{item.title}</strong><small>{item.detail}</small></div>
                  <div className="export-name"><input value={nameOf(item)} onChange={(event) => setNames((previous) => ({ ...previous, [item.id]: event.target.value }))} onBlur={(event) => setNames((previous) => ({ ...previous, [item.id]: cleanName(event.target.value) }))} aria-label={`File name for ${item.title}`} spellCheck={false} /><span>.{f.ext}</span></div>
                  {item.formats.length > 1 ? <select className="export-format" value={f.id} onChange={(event) => setFormats((previous) => ({ ...previous, [item.id]: event.target.value }))} aria-label={`Format for ${item.title}`}>{item.formats.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}</select>
                    : <span className="export-format is-single">{f.label}</span>}
                  <span className={`export-size${item.bytes == null ? " made" : ""}`}>{item.bytes == null ? "Made on download" : bytes(item.bytes)}</span>
                </li>; })}</ul>
            </div> : null; })}
        </section>
      </div>

      <footer className={`export-bar${chosen.length ? " has-items" : ""}`}>
        <div className="export-count"><strong>{chosen.length}</strong> selected<span>{knownBytes ? bytes(knownBytes) : "—"}{unknown ? " + files made on download" : ""}</span></div>
        <div className="segmented" role="group" aria-label="Download as">
          <button aria-pressed={mode === "zip"} onClick={() => setMode("zip")}><Icon name="folder" size={14} />One .zip</button>
          <button aria-pressed={mode === "files"} onClick={() => setMode("files")}><Icon name="file" size={14} />Separate files</button>
        </div>
        <button className="button primary" disabled={!chosen.length} onClick={review}><Icon name="download" size={16} />Review & download</button>
      </footer>
    </main>

    <dialog ref={dialog} className="studio-dialog export-dialog" onCancel={(event) => { if (running) event.preventDefault(); }}>
      <div className="dialog-heading"><div><span className="eyebrow">{finished ? (failed ? "Finished with problems" : "Done") : "Check before downloading"}</span><h2>{finished ? `${doneCount} of ${chosen.length} ready` : `${chosen.length} file${chosen.length === 1 ? "" : "s"}, ${mode === "zip" ? "in one .zip" : "as separate downloads"}`}</h2><p>{mode === "zip" ? "Everything below goes into one archive." : "Your browser may ask once to allow several downloads."}</p></div><button className="icon-button" aria-label="Close" disabled={running} onClick={() => dialog.current?.close()}><Icon name="close" /></button></div>
      <div className="dialog-body">
        {mode === "zip" && <label className="field export-zipname">Archive name<div className="export-name"><input value={zipName} onChange={(event) => setZipName(event.target.value)} disabled={running} spellCheck={false} /><span>.zip</span></div></label>}
        <ol className="export-review">{finalNames().map(({ item, format, name }) => { const s = steps[item.id];
          return <li key={item.id} className={s ? `is-${s.step}` : ""}>
            <span className="export-state">{s?.step === "done" ? <Icon name="check" size={14} /> : s?.step === "failed" ? <Icon name="alert" size={14} /> : s?.step === "working" ? <span className="spinner" /> : <Icon name={item.icon} size={14} />}</span>
            <div><strong>{name}</strong><small>{s?.step === "failed" ? s.note : `${item.title} · ${format.label}${s?.note ? ` · ${s.note}` : item.bytes != null ? ` · ${bytes(item.bytes)}` : ""}`}</small></div>
          </li>; })}</ol>
      </div>
      <div className="dialog-footer"><span><Icon name="shield" size={14} />Made on this machine; nothing is uploaded.</span>
        {finished ? <button className="button primary" onClick={() => dialog.current?.close()}>Close</button>
          : <button className="button primary" disabled={running || !chosen.length} onClick={() => void download()}>{running ? <><span className="spinner" />Preparing…</> : <><Icon name="download" size={16} />Download</>}</button>}
      </div>
    </dialog>
  </Studio>;
}
