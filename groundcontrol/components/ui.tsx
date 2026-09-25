import Link from "next/link";

/**
 * Console primitives. Deliberately few: a panel, a status chip, a reading, a bar.
 * Status is always hue + word + glyph so colour-blind reviewers read the same page.
 */

export const TONE: Record<string, { text: string; dot: string; bg: string }> = {
  verified: { text: "text-[--color-verified]", dot: "bg-[--color-verified]", bg: "bg-[--color-verified]/10" },
  signal: { text: "text-[--color-signal]", dot: "bg-[--color-signal]", bg: "bg-[--color-signal]/10" },
  refused: { text: "text-[--color-refused]", dot: "bg-[--color-refused]", bg: "bg-[--color-refused]/10" },
  pending: { text: "text-[--color-pending]", dot: "bg-[--color-pending]", bg: "bg-[--color-pending]/10" },
  mechanism: { text: "text-[--color-mechanism]", dot: "bg-[--color-mechanism]", bg: "bg-[--color-mechanism]/10" },
  mute: { text: "text-[--color-ink-mute]", dot: "bg-[--color-ink-mute]", bg: "bg-white/5" },
};

export function toneFor(status: string) {
  const key = status.toLowerCase();
  if (["measured", "delivered", "complete", "ready", "built", "fulfilled", "meets_target", "aligned", "evaluated", "prepared"].some((s) => key.includes(s))) return "verified";
  if (["not_delivered", "refused", "fail", "missing", "absent", "exceeds", "invalid", "not-fulfilled", "not-built"].some((s) => key.includes(s))) return "refused";
  if (["mechanism"].includes(key)) return "mechanism";
  if (["partial", "pending", "running", "needs"].some((s) => key.includes(s))) return "pending";
  if (["not_evaluated", "unknown", "not available"].some((s) => key.includes(s))) return "mute";
  return "mute";
}

export function Panel({ title, hint, right, children, className = "" }: {
  title?: string; hint?: string; right?: React.ReactNode; children: React.ReactNode; className?: string;
}) {
  return (
    <section className={`panel relative ${className}`}>
      {(title || right) && (
        <header className="flex items-baseline justify-between gap-4 border-b px-5 py-3">
          <div>
            <h2 className="text-[13px] font-medium tracking-tight text-ink">{title}</h2>
            {hint && <p className="label mt-1 normal-case tracking-normal text-[--color-ink-mute]">{hint}</p>}
          </div>
          {right}
        </header>
      )}
      <div className="px-5 py-4">{children}</div>
    </section>
  );
}

export function Chip({ status, label }: { status: string; label?: string }) {
  const tone = TONE[toneFor(status)];
  return (
    <span className={`inline-flex items-center gap-1.5 border px-2 py-[3px] text-[10px] uppercase tracking-[0.12em] ${tone.text} border-current/30`}>
      <span className={`size-1.5 rounded-full ${tone.dot}`} aria-hidden />
      {label ?? status.replace(/_/g, " ")}
    </span>
  );
}

export function Reading({ label, value, unit, status, hint, size = "md" }: {
  label: string; value: React.ReactNode; unit?: string; status?: string;
  hint?: string; size?: "md" | "lg";
}) {
  return (
    <div className="flex flex-col gap-2">
      <span className="label">{label}</span>
      <div className="flex items-baseline gap-1.5">
        <span className={`num font-light leading-none ${size === "lg" ? "text-[40px]" : "text-[26px]"} text-ink`}>
          {value}
        </span>
        {unit && <span className="num text-[11px] text-[--color-ink-mute]">{unit}</span>}
      </div>
      {status && <Chip status={status} />}
      {hint && <p className="text-[11px] leading-relaxed text-[--color-ink-mute]">{hint}</p>}
    </div>
  );
}

export function Bar({ fraction, tone = "signal" }: { fraction: number; tone?: string }) {
  const width = Math.max(0, Math.min(1, Number.isFinite(fraction) ? fraction : 0)) * 100;
  return (
    <div className="h-[3px] w-full bg-white/5" role="presentation">
      <div className={`h-full ${TONE[tone]?.dot ?? TONE.signal.dot} transition-[width] duration-500`} style={{ width: `${width}%` }} />
    </div>
  );
}

export function Row({ cells, head = false }: { cells: React.ReactNode[]; head?: boolean }) {
  return (
    <div className={`grid gap-3 border-b px-1 py-2.5 last:border-0 ${head ? "label" : "text-[12.5px]"}`}
         style={{ gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))" }}>
      {cells.map((cell, index) => <div key={index} className="min-w-0">{cell}</div>)}
    </div>
  );
}

export function SectionTitle({ kicker, title, lede }: { kicker: string; title: string; lede?: string }) {
  return (
    <div className="mb-6">
      <span className="label text-[--color-signal]">{kicker}</span>
      <h1 className="mt-2 text-[26px] font-medium tracking-tight text-ink">{title}</h1>
      {lede && <p className="mt-2 max-w-[68ch] text-[13px] leading-relaxed text-[--color-ink-dim]">{lede}</p>}
    </div>
  );
}

export function OffLine({ error }: { error: string }) {
  return (
    <div className="panel border-[--color-refused]/40 bg-[--color-refused]/5 p-5">
      <div className="flex items-center gap-2">
        <span className="size-2 rounded-full bg-[--color-refused]" />
        <span className="label text-[--color-refused]">Backend unreachable</span>
      </div>
      <p className="mt-3 text-[13px] text-[--color-ink-dim]">
        The console reads a live reconstruction backend; nothing here is mocked. Start it with
      </p>
      <pre className="num mt-2 overflow-x-auto border bg-black/40 p-3 text-[11px] text-[--color-signal]">
        .venv/Scripts/python.exe _serve.py 8137 .
      </pre>
      <p className="num mt-3 text-[11px] text-[--color-ink-mute]">{error}</p>
    </div>
  );
}

const NAV = [
  { href: "/", label: "Mission", hint: "targets and ledger" },
  { href: "/pipeline", label: "Pipeline", hint: "run, tail, timeline" },
  { href: "/survey", label: "Survey", hint: "inputs → evaluate" },
  { href: "/deliverables", label: "Deliverables", hint: "six formats" },
  { href: "/challenges", label: "Challenges", hint: "the eight" },
  { href: "/applications", label: "Applications", hint: "eight verticals" },
];

export function Shell({ children, scene }: { children: React.ReactNode; scene?: string }) {
  return (
    <div className="min-h-screen">
      <div className="hairline-grid pointer-events-none fixed inset-0 opacity-[0.35]" aria-hidden />
      <header className="sticky top-0 z-20 border-b bg-[--color-canvas]/85 backdrop-blur">
        <div className="mx-auto flex max-w-[1320px] flex-wrap items-center gap-x-6 gap-y-1 px-6 py-3">
          <Link href="/" className="flex items-center gap-3">
            <span className="relative grid size-7 place-items-center border border-[--color-signal]/50">
              <span className="absolute size-1.5 rounded-full bg-[--color-signal]" />
              <span className="absolute inset-1 rounded-full border border-[--color-signal]/30" />
            </span>
            <span className="text-[13px] font-medium tracking-[0.2em] text-ink">GROUND CONTROL</span>
          </Link>
          <nav className="-mx-1 order-3 w-full overflow-x-auto px-1 lg:order-none lg:mx-0 lg:w-auto lg:overflow-visible" aria-label="Console">
            <div className="flex w-max items-center gap-1 lg:w-auto">
            {NAV.map((item) => (
              <Link key={item.href} href={item.href}
                    className="group border border-transparent px-3 py-1.5 transition-colors hover:border-hairline">
                <span className="block whitespace-nowrap text-[12.5px] text-[--color-ink-dim] group-hover:text-ink">{item.label}</span>
                <span className="label block whitespace-nowrap text-[9px] normal-case tracking-normal">{item.hint}</span>
              </Link>
            ))}
            </div>
          </nav>
          <div className="ml-auto flex items-center gap-3">
            {scene && (
              <span className="num hidden whitespace-nowrap text-[11px] text-[--color-ink-mute] 2xl:inline">scene <span className="text-ink">{scene}</span></span>
            )}
            <span className="flex items-center gap-1.5 border border-[--color-signal]/30 px-2 py-1">
              <span className="size-1.5 rounded-full bg-[--color-signal]" />
              <span className="label text-[--color-signal]">GPU gated</span>
            </span>
          </div>
        </div>
      </header>
      <main className="relative z-10 mx-auto max-w-[1320px] px-6 py-8">{children}</main>
      <footer className="relative z-10 mx-auto max-w-[1320px] px-6 pb-10 pt-4">
        <p className="text-[11px] leading-relaxed text-[--color-ink-mute]">
          SIH26158 · Single-Pass Drone Video to Accurate 3D Model. Every reading is labelled
          measured, projected from measured rates, or synthetic fixture. A module existing is never
          shown as a module running.
        </p>
      </footer>
    </div>
  );
}
