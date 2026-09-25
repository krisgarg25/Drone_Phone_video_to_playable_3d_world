import { Chip, Panel, SectionTitle, Shell } from "@/components/ui";
import { VERTICALS } from "@/lib/data";

const STATE_TONE: Record<string, string> = { built: "verified", "needs-run": "pending", "not-built": "refused" };
const STATE_LABEL: Record<string, string> = { built: "Tool works", "needs-run": "Needs a real run", "not-built": "Not built" };

export default function Applications() {
  const built = VERTICALS.filter((v) => v.state === "built").length;
  const needsRun = VERTICALS.filter((v) => v.state === "needs-run").length;
  const notBuilt = VERTICALS.filter((v) => v.state === "not-built").length;

  return (
    <Shell>
      <SectionTitle kicker="Potential applications"
        title="Eight verticals, and the tool each one would actually use"
        lede="A vertical is a set of tools on top of the geometry. Where the maths exists it is marked as working; where it exists but no qualifying scene has produced numbers it says so; where nothing exists — semantic extraction, furniture placement, pre/post registration — the panel says not built rather than showing a placeholder." />

      <div className="mb-4 flex flex-wrap gap-2 text-[12px]">
        {[["built", built], ["needs-run", needsRun], ["not-built", notBuilt]].map(([state, count]) => (
          <div key={String(state)} className="flex items-center gap-2 border px-3 py-1.5">
            <Chip status={STATE_TONE[String(state)]} label={STATE_LABEL[String(state)]} />
            <span className="num text-ink">{String(count)} / 8</span>
          </div>
        ))}
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        {VERTICALS.map((vertical) => (
          <article key={vertical.id} className="panel p-5">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-[14px] font-medium text-ink">{vertical.name}</h2>
                <p className="mt-1 text-[12px] leading-relaxed text-[--color-ink-mute]">{vertical.blurb}</p>
              </div>
              <Chip status={STATE_TONE[vertical.state]} label={STATE_LABEL[vertical.state]} />
            </div>
            <ul className="mt-4 space-y-2">
              {vertical.tools.map((tool) => (
                <li key={tool.name} className="flex items-start gap-3 border-t pt-2.5 text-[12.5px]">
                  <span className={`mt-[6px] size-1.5 shrink-0 rounded-full ${
                    tool.state === "built" ? "bg-[--color-verified]" : tool.state === "needs-run" ? "bg-[--color-pending]" : "bg-[--color-refused]"}`} />
                  <div className="min-w-0">
                    <p className="text-ink">{tool.name}</p>
                    <p className="text-[11px] leading-relaxed text-[--color-ink-mute]">{tool.note}</p>
                  </div>
                </li>
              ))}
            </ul>
          </article>
        ))}
      </div>

      <Panel title="The one scope decision left"
        hint="Everything else is a tool on geometry the system already produces.">
        <p className="max-w-[82ch] text-[12.5px] leading-relaxed text-[--color-ink-dim]">
          Urban planning, and the requirements for buildings, roads and vegetation, need
          <span className="text-ink"> semantic extraction</span> — a segmentation pass that turns a
          point cloud into labelled objects. Nothing in this repository does that today. The
          measurement verticals (inspection, archaeology, damage volumes, walkable rehearsal) do not
          need it: they need a click path onto maths that is already written and tested. That is the
          order to build in.
        </p>
      </Panel>
    </Shell>
  );
}
