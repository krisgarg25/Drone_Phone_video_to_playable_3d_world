## What and why

<!-- One paragraph: what changes for the operator, and which playbook item (e.g. INF-02, URB-08) it closes. -->

## Evidence

<!-- Numbers from a test or a run, not adjectives. Which suite, which scene, which known answer. -->

## Production checklist (docs/APPLICATION_PLAYBOOKS.md §5)

**Honesty**
- [ ] Measured vs proposed vs inferred vs unobserved is visible in the UI and in every export this touches
- [ ] No accuracy claimed from a fit, a render or our own output; unverified results are labelled as such
- [ ] A refused step says what was refused and why (no silent fallback)

**Experience**
- [ ] Loading, empty, error and "not measurable" states, each with a next action
- [ ] Undo/redo or a clear way back for every edit; nothing lost on refresh
- [ ] Units and coordinate frame shown wherever a number is shown
- [ ] Works at 1366×768 and in the dark theme; keyboard reachable

**Reliability and security**
- [ ] Inputs validated server-side; a corrupt file is refused with a reason, not a crash
- [ ] Writes pass the loopback/Origin guard (or a documented, token-scoped exception)
- [ ] Works offline: no CDN, no external tiles, no telemetry leaving the machine

**Quality**
- [ ] Unit tests for new modules against known answers; `tests/check_all.py` clean
- [ ] Browser check of the golden path for the touched tab
- [ ] `docs/GAPS_AND_OPTIMIZATIONS.md` updated: what is built, what is verified, what is not
