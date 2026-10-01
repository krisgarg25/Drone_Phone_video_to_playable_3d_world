import { Studio } from "@/components/studio";
import { CHALLENGES, VERDICT_LABEL, VERDICT_TONE, type Verdict } from "@/lib/data";

const ORDER: Verdict[] = ["fulfilled", "mechanism", "partial", "not-fulfilled", "absent"];

export default function Challenges() {
  const tally = CHALLENGES.reduce<Record<string, number>>((total, item) => {
    total[item.verdict] = (total[item.verdict] ?? 0) + 1;
    return total;
  }, {});

  return (
    <Studio>
      <main className="library-main">
        <section className="hub-hero">
          <div>
            <div className="eyebrow">SIH 26158 · Key challenges</div>
            <h1>Eight challenges.<br /><em>Honestly graded.</em></h1>
          </div>
          <p>Each verdict is backed by what the system measured, not what it claims. A module that runs and passes its tests is marked as a mechanism until data that qualifies has shown its benefit.</p>
        </section>

        <div className="tally">
          {ORDER.filter((v) => tally[v]).map((verdict) => (
            <div key={verdict}><span className={`pill tone-${VERDICT_TONE[verdict]}`}>{VERDICT_LABEL[verdict]}</span><b>{tally[verdict]}</b></div>
          ))}
        </div>

        <div className="challenge-grid">
          {CHALLENGES.map((item, index) => (
            <article key={item.id} className="challenge" style={{ animationDelay: `${index * 0.04}s` }}>
              <div className="challenge-head">
                <div><span className="challenge-num">({item.id})</span><h2>{item.title}</h2></div>
                <span className={`pill tone-${VERDICT_TONE[item.verdict]}`}>{VERDICT_LABEL[item.verdict]}</span>
              </div>
              <p className="challenge-evidence"><b>Evidence</b>{item.evidence}</p>
              <p className="challenge-limit"><b>Limit</b>{item.limit}</p>
            </article>
          ))}
        </div>

        <section className="callout">
          <h3>Why &ldquo;mechanism&rdquo; and &ldquo;fulfilled&rdquo; are kept apart</h3>
          <p>The dynamic-object case carries a measured trade: masking <b>4.85%</b> of pixels cost <b>18.4%</b> of sparse tracks and improved mean reprojection error by <b>7.8%</b>, with no loss of registered cameras. The others rest on code that executes and tests that pass — real, but not yet evidence about a reconstruction. The gap is one qualifying dataset wide.</p>
        </section>
      </main>
    </Studio>
  );
}
