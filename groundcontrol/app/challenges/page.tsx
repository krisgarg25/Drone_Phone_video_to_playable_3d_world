import { Chip, Panel, SectionTitle, Shell } from "@/components/ui";
import { CHALLENGES, VERDICT_LABEL, VERDICT_TONE } from "@/lib/data";

export default function Challenges() {
  const tally = CHALLENGES.reduce<Record<string, number>>((total, item) => {
    total[item.verdict] = (total[item.verdict] ?? 0) + 1;
    return total;
  }, {});

  return (
    <Shell>
      <SectionTitle kicker="Problem statement · key challenges"
        title="The eight challenges, honestly graded"
        lede="One of the eight is fulfilled with a measurement behind it. Five are mechanism-complete and waiting on qualifying footage. One is partial by deliberate choice, and one is not met on this hardware. Anything else would be a claim, not a result." />

      <div className="mb-4 flex flex-wrap gap-2">
        {Object.entries(tally).map(([verdict, count]) => (
          <div key={verdict} className="flex items-center gap-2 border px-3 py-1.5">
            <Chip status={VERDICT_TONE[verdict as keyof typeof VERDICT_TONE]} label={VERDICT_LABEL[verdict as keyof typeof VERDICT_LABEL]} />
            <span className="num text-[13px] text-ink">{count}</span>
          </div>
        ))}
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        {CHALLENGES.map((item) => (
          <article key={item.id} className="panel flex flex-col gap-3 p-5">
            <div className="flex items-start justify-between gap-4">
              <div className="flex items-baseline gap-3">
                <span className="num text-[11px] text-[--color-signal]">({item.id})</span>
                <h2 className="text-[14px] font-medium leading-snug text-ink">{item.title}</h2>
              </div>
              <Chip status={VERDICT_TONE[item.verdict]} label={VERDICT_LABEL[item.verdict]} />
            </div>
            <div className="space-y-2.5 text-[12.5px] leading-relaxed">
              <p className="text-[--color-ink-dim]"><span className="label mr-2">Evidence</span>{item.evidence}</p>
              <p className="border-l-2 border-[--color-refused]/40 pl-3 text-[--color-ink-mute]">
                <span className="label mr-2">Limit</span>{item.limit}
              </p>
            </div>
          </article>
        ))}
      </div>

      <Panel title="Why the split between mechanism and fulfilled matters"
        hint="A module that runs and is tested is not a module whose benefit has been measured on data that qualifies.">
        <p className="max-w-[80ch] text-[12.5px] leading-relaxed text-[--color-ink-dim]">
          The dynamic-object case is the only one with a number attached: masking 4.85% of pixels
          cost 18.4% of the sparse tracks and improved mean reprojection error by 7.8%, with no
          loss of registered cameras. That is a measured trade. The other seven currently rest on
          code that executes and tests that pass — real, but not evidence about a reconstruction.
          The gap between them is one qualifying dataset wide.
        </p>
      </Panel>
    </Shell>
  );
}
