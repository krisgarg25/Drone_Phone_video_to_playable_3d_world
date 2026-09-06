/* Combat HUD. Injects its own CSS + DOM so the viewer page stays untouched. */

const CSS = `
:root {
  --cg-ink: #eaf2fb;
  --cg-dim: #8fa3b8;
  --cg-accent: #63b3ed;
  --cg-good: #48bb78;
  --cg-warn: #f6ad55;
  --cg-bad: #fc8181;
  --cg-panel: rgba(11, 15, 23, .82);
  --cg-line: rgba(255, 255, 255, .12);
  --cg-radius: 10px;
  --cg-gap: 8px;
  --cg-font: 12px;
  --cg-cross: 22px;
}
.cg-hud { position: fixed; inset: 0; z-index: 20; pointer-events: none;
  font-family: system-ui, -apple-system, 'Segoe UI', 'JetBrains Mono', monospace;
  color: var(--cg-ink); font-size: var(--cg-font); }
.cg-cross { position: absolute; left: 50%; top: 50%; width: var(--cg-cross); height: var(--cg-cross);
  margin: calc(var(--cg-cross) / -2); transition: transform .06s linear; }
.cg-cross i { position: absolute; background: rgba(255,255,255,.9); box-shadow: 0 0 3px rgba(0,0,0,.9); }
.cg-cross .v { left: 50%; top: 0; width: 1.5px; height: 38%; }
.cg-cross .v2 { left: 50%; bottom: 0; width: 1.5px; height: 38%; }
.cg-cross .h { top: 50%; left: 0; height: 1.5px; width: 38%; }
.cg-cross .h2 { top: 50%; right: 0; height: 1.5px; width: 38%; }
.cg-cross .dot { left: 50%; top: 50%; width: 2px; height: 2px; margin: -1px; border-radius: 50%; }
.cg-hit { position: absolute; left: 50%; top: 50%; width: 30px; height: 30px; margin: -15px; opacity: 0; }
.cg-hit b { position: absolute; left: 50%; top: 50%; width: 2.5px; height: 10px; margin: -5px 0 0 -1.25px;
  background: var(--cg-ink); }
.cg-hit.show { animation: cg-pop .22s ease-out; }
.cg-hit.kill b { background: var(--cg-bad); }
@keyframes cg-pop { 0% { opacity: 1; transform: scale(.5) rotate(45deg); } 100% { opacity: 0; transform: scale(1.5) rotate(45deg); } }
.cg-vig { position: absolute; inset: 0; opacity: 0; transition: opacity .18s;
  background: radial-gradient(ellipse at center, transparent 42%, rgba(190, 20, 30, .62) 100%); }
.cg-vig.low { animation: cg-throb 1.5s ease-in-out infinite; }
@keyframes cg-throb { 0%,100% { opacity: .28; } 50% { opacity: .5; } }
.cg-bar { position: absolute; left: 16px; bottom: 16px; min-width: 240px;
  background: var(--cg-panel); backdrop-filter: blur(12px); border: 1px solid var(--cg-line);
  border-radius: var(--cg-radius); padding: 10px 12px; display: flex; flex-direction: column; gap: 6px; }
.cg-row { display: flex; align-items: center; gap: var(--cg-gap); justify-content: space-between; }
.cg-label { color: var(--cg-dim); letter-spacing: .06em; font-size: 10px; text-transform: uppercase; }
.cg-track { position: relative; width: 150px; height: 7px; border-radius: 4px; background: rgba(255,255,255,.1); overflow: hidden; }
.cg-fill { position: absolute; inset: 0; transform-origin: left; transition: transform .12s ease-out; background: var(--cg-good); }
.cg-fill.warn { background: var(--cg-warn); } .cg-fill.bad { background: var(--cg-bad); }
.cg-ammo { font-variant-numeric: tabular-nums; font-size: 20px; font-weight: 700; }
.cg-ammo small { font-size: 11px; color: var(--cg-dim); font-weight: 500; }
.cg-ammo.dry { color: var(--cg-bad); }
.cg-top { position: absolute; left: 50%; top: 14px; transform: translateX(-50%); text-align: center;
  background: var(--cg-panel); backdrop-filter: blur(12px); border: 1px solid var(--cg-line);
  border-radius: var(--cg-radius); padding: 7px 14px; display: flex; gap: 14px; align-items: baseline; }
.cg-top b { color: var(--cg-accent); font-variant-numeric: tabular-nums; }
.cg-feed { position: absolute; right: 16px; top: 60px; display: flex; flex-direction: column-reverse; gap: 4px;
  align-items: flex-end; }
.cg-feed div { background: var(--cg-panel); border: 1px solid var(--cg-line); border-radius: 7px;
  padding: 4px 9px; opacity: 0; animation: cg-in .2s forwards; }
.cg-feed div.out { animation: cg-out .4s forwards; }
@keyframes cg-in { to { opacity: 1; } } @keyframes cg-out { to { opacity: 0; transform: translateX(12px); } }
.cg-status { position: absolute; left: 50%; bottom: 22px; transform: translateX(-50%);
  background: rgba(190, 30, 40, .9); border: 1px solid rgba(255,180,180,.45); color: #fff;
  padding: 8px 14px; border-radius: 8px; font-weight: 600; max-width: 70ch; }
.cg-status.ok { background: rgba(20, 60, 40, .9); border-color: rgba(120, 220, 160, .4); }
.cg-dbg { position: absolute; right: 16px; bottom: 16px; background: var(--cg-panel);
  backdrop-filter: blur(12px); border: 1px solid var(--cg-line); border-radius: var(--cg-radius);
  padding: 9px 11px; min-width: 320px; pointer-events: auto; }
.cg-dbg h4 { margin: 0 0 6px; font-size: 10px; letter-spacing: .1em; text-transform: uppercase;
  color: var(--cg-dim); display: flex; justify-content: space-between; gap: 10px; }
.cg-dbg table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
.cg-dbg td { padding: 2px 6px 2px 0; font-size: 11px; border-top: 1px solid rgba(255,255,255,.05); }
.cg-dbg td:first-child { color: var(--cg-dim); }
.cg-dbg .st { font-weight: 700; }
.cg-dbg .st.HOLD, .cg-dbg .st.AMBUSH { color: var(--cg-accent); }
.cg-dbg .st.ENGAGE { color: var(--cg-bad); }
.cg-dbg .st.FLANK, .cg-dbg .st.RELOCATE { color: var(--cg-warn); }
.cg-dbg .st.SEARCH { color: var(--cg-good); }
.cg-dbg .st.DEAD { color: #5a6a7a; }
.cg-btn { pointer-events: auto; cursor: pointer; background: rgba(255,255,255,.1); color: var(--cg-ink);
  border: 1px solid var(--cg-line); border-radius: 6px; font: inherit; font-size: 10.5px; padding: 2px 7px; }
.cg-btn:hover { background: rgba(255,255,255,.2); }
.cg-help { position: absolute; left: 16px; top: 16px; background: var(--cg-panel);
  border: 1px solid var(--cg-line); border-radius: var(--cg-radius); padding: 8px 11px;
  color: var(--cg-dim); line-height: 1.65; }
.cg-help b { color: var(--cg-ink); }
`;

function el(tag, cls, html) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
}

export class CombatHUD {
  constructor() {
    if (!document.getElementById("cg-style")) {
      const style = el("style");
      style.id = "cg-style";
      style.textContent = CSS;
      document.head.appendChild(style);
    }
    this.root = el("div", "cg-hud");
    this.cross = el("div", "cg-cross", '<i class="v"></i><i class="v2"></i><i class="h"></i><i class="h2"></i><i class="dot"></i>');
    this.hitmark = el("div", "cg-hit", "<b></b><b></b><b></b><b></b>");
    this.vig = el("div", "cg-vig");
    this.top = el("div", "cg-top");
    this.feedEl = el("div", "cg-feed");
    this.bar = el("div", "cg-bar");
    this.dbg = el("div", "cg-dbg");
    this.help = el("div", "cg-help");
    this.status = null;
    this.hitmark.children[0].style.transform = "translateY(-9px)";
    this.hitmark.children[1].style.transform = "translateY(9px)";
    this.hitmark.children[2].style.transform = "translate(-9px) rotate(90deg)";
    this.hitmark.children[3].style.transform = "translate(9px) rotate(90deg)";
    this.root.append(this.vig, this.cross, this.hitmark, this.top, this.feedEl, this.bar, this.dbg, this.help);
    document.body.appendChild(this.root);
    this._buildBar();
    this._buildDbg();
    this._buildTop();
    this.dbg.hidden = true;
    this.help.innerHTML =
      "<b>LMB</b> fire &nbsp; <b>R</b> reload &nbsp; <b>Shift</b> sprint &nbsp; <b>Space</b> jump &nbsp; " +
      "<b>H</b> AI debug &nbsp; <b>Esc</b> release mouse<br>" +
      "Click the scene to lock the mouse. Sprinting and firing are loud — bots run toward noise.";
    this._lastHit = 0;
  }

  _buildBar() {
    const hpRow = el("div", "cg-row");
    const track = el("div", "cg-track");
    this.hpFill = el("div", "cg-fill");
    track.appendChild(this.hpFill);
    this.hpText = el("b");
    this.hpText.textContent = "100";
    hpRow.append(el("span", "cg-label", "Health"), track, this.hpText);

    this.ammoText = el("div", "cg-ammo");
    const ammoRow = el("div", "cg-row");
    ammoRow.append(el("span", "cg-label", "Magazine"), this.ammoText);
    this.bar.append(hpRow, ammoRow);
    this._setAmmo(0, 0);
  }

  _setAmmo(mag, reserve) {
    this.ammoText.replaceChildren(
      document.createTextNode(String(mag)),
      el("small", "", ` / ${reserve}`),
    );
    this.ammoText.classList.toggle("dry", mag === 0);
  }

  _buildTop() {
    const field = (label) => {
      const wrap = el("span");
      wrap.append(document.createTextNode(`${label} `));
      const b = el("b");
      b.textContent = "0";
      wrap.appendChild(b);
      this.top.appendChild(wrap);
      return b;
    };
    this.contacts = field("Contacts");
    this.kills = field("Downed");
    this.threat = field("Threat");
    this.threat.textContent = "—";
  }

  _buildDbg() {
    this.dbg.innerHTML =
      "<h4><span>Bot internals</span><span><button class=\"cg-btn\" data-act=\"copy\">copy state</button> " +
      "<button class=\"cg-btn\" data-act=\"close\">close</button></span></h4>" +
      "<table><tbody></tbody></table>";
    this.dbgBody = this.dbg.querySelector("tbody");
    this.dbg.addEventListener("click", (e) => {
      const act = e.target.dataset?.act;
      if (act === "close") this.setDebug(false);
      if (act === "copy") this.copyState();
    });
  }

  /** @param {() => object} provider snapshot used by the copy-state probe. */
  bindProbe(provider) {
    this.probe = provider;
  }

  copyState() {
    if (!this.probe) return;
    const text = JSON.stringify(this.probe(), null, 2);
    navigator.clipboard?.writeText(text).then(
      () => this.toast("AI state copied to clipboard", true),
      () => this.toast("Clipboard blocked — state logged to console", false),
    );
    console.log(text);
  }

  setSpread(px) {
    const s = 1 + Math.max(0, px) / 26;
    this.cross.style.transform = `scale(${s.toFixed(3)})`;
    this.cross.style.opacity = document.pointerLockElement ? "1" : "0.25";
  }

  hitmarker(kill = false) {
    const now = performance.now();
    if (now - this._lastHit < 40) return;
    this._lastHit = now;
    this.hitmark.classList.toggle("kill", kill);
    this.hitmark.classList.remove("show");
    void this.hitmark.offsetWidth;
    this.hitmark.classList.add("show");
  }

  feed(text, tone = "info") {
    const row = el("div");
    row.textContent = text;
    row.style.borderColor = tone === "kill" ? "rgba(252,129,129,.5)" : tone === "warn" ? "rgba(246,173,85,.5)" : "rgba(255,255,255,.12)";
    this.feedEl.prepend(row);
    setTimeout(() => row.classList.add("out"), 2600);
    setTimeout(() => row.remove(), 3100);
    while (this.feedEl.children.length > 6) this.feedEl.lastChild.remove();
  }

  toast(text, ok = false) {
    this.error(text, ok ? "ok" : "info", 2200);
  }

  error(text, kind = "err", autoHide = 0) {
    this.status?.remove();
    this.status = el("div", `cg-status${kind === "ok" ? " ok" : ""}`);
    this.status.textContent = text;
    this.root.appendChild(this.status);
    if (autoHide) setTimeout(() => { this.status?.remove(); this.status = null; }, autoHide);
  }

  clearError() {
    this.status?.remove();
    this.status = null;
  }

  render(s) {
    const hp = Math.max(0, s.health) / s.healthMax;
    this.hpFill.style.transform = `scaleX(${hp})`;
    this.hpFill.className = `cg-fill${hp < 0.3 ? " bad" : hp < 0.6 ? " warn" : ""}`;
    this.hpText.textContent = String(Math.ceil(Math.max(0, s.health)));
    this._setAmmo(s.ammo, s.reserve);
    this.contacts.textContent = String(s.alive);
    this.kills.textContent = String(s.kills);
    this.threat.textContent = s.threat ? `${s.threat.label} ${s.threat.dist.toFixed(0)}m` : "clear";
    this.threat.style.color = s.threat?.firing ? "var(--cg-bad)" : s.threat ? "var(--cg-warn)" : "var(--cg-ink)";
    this.vig.style.opacity = String(Math.min(0.85, s.pain));
    this.vig.classList.toggle("low", hp < 0.3 && s.alive > 0);
  }

  setBots(bots) {
    if (this.dbg.hidden) return;
    this.dbgBody.replaceChildren(...bots.map((b) => {
      const tr = document.createElement("tr");
      const cell = (text, cls) => {
        const td = document.createElement("td");
        td.textContent = text;
        if (cls) td.className = cls;
        tr.appendChild(td);
      };
      cell(`#${b.id}`);
      cell(b.state, "st " + b.state);
      cell(`aw ${b.awareness.toFixed(2)}`);
      cell(`${b.dist.toFixed(0)}m`);
      cell(b.path);
      cell(b.spot || "—");
      return tr;
    }));
    if (!bots.length) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.textContent = "no bots";
      tr.appendChild(td);
      this.dbgBody.appendChild(tr);
    }
  }

  toggleDebug(force) {
    this.dbg.hidden = force !== undefined ? !force : !this.dbg.hidden;
    return !this.dbg.hidden;
  }
}
