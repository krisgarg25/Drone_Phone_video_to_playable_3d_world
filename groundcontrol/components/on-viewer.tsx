"use client";

import { createContext, useContext, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Icon } from "./studio-icons";

/** The element over the 3D view that click-to-pick guides render into. */
export const ViewerSlot = createContext<HTMLElement | null>(null);

/**
 * Shows a tool's step-by-step guide on the 3D view, where the user is clicking,
 * instead of in the side panel. Falls back to rendering in place.
 */
export function OnViewer({ children, kicker = "Click on the model" }: { children: React.ReactNode; kicker?: string }) {
  const slot = useContext(ViewerSlot);
  const body = <div className="pick-guide"><span className="pick-kicker"><Icon name="cursor" size={14} />{kicker}</span>{children}</div>;
  return slot ? createPortal(body, slot) : body;
}

/** A small (i) that opens a plain-language note. */
export function InfoTip({ children, label = "More info", align = "right" }: { children: React.ReactNode; label?: string; align?: "left" | "right" }) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (event: PointerEvent) => { if (!box.current?.contains(event.target as Node)) setOpen(false); };
    const esc = (event: KeyboardEvent) => { if (event.key === "Escape") setOpen(false); };
    window.addEventListener("pointerdown", close); window.addEventListener("keydown", esc);
    return () => { window.removeEventListener("pointerdown", close); window.removeEventListener("keydown", esc); };
  }, [open]);
  return <span className="info-tip" ref={box}>
    <button type="button" className={`info-tip-button${open ? " is-open" : ""}`} aria-label={label} aria-expanded={open} onClick={() => setOpen(!open)}><Icon name="info" size={15} /></button>
    {open && <span className={`info-pop align-${align}`} role="dialog" aria-label={label}>{children}</span>}
  </span>;
}
