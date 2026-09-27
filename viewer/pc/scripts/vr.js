/* WebXR (VR) rehearsal (RH-7). Inert unless the browser offers immersive-vr.

A rig entity stands at the player's feet and turns with the player's yaw; the camera is
parented to it while a session runs, so the headset moves the eyes around the body and
the thumbsticks move the body over the scan:

* left stick  - smooth locomotion relative to where the HEAD looks (dead zone 0.15),
                at walking speed, through the same capsule and navmesh as the desktop;
* right stick - snap turn 30 degrees (comfort: no smooth yaw), re-armed below 0.3;
* right trigger (held)  - fire along the right controller's pointer ray (tracked-pointer
                input); a controller without tracking falls back to the head's aim;
* the scan's own "unobserved" hatching and the rehearsal labels are unchanged.

What is NOT verified here: frame rate on a headset (the 72 fps target needs the
decimated-mesh / splat-LOD path of RH-7), and controller models. The session is offered
only where ``navigator.xr`` says immersive-vr is supported (Quest browser, PC VR over
HTTPS - the local ``_cert.pem`` serves this).
*/
import { Entity, XRSPACE_LOCALFLOOR, XRTYPE_VR } from "playcanvas";
import { snapTurn, stickToMove, yawOf } from "./vr_input.js";

export function installVr({ app, camera, canvas, onTrigger, onStatus }) {
  const xr = app.xr;
  const state = { active: false, move: { dx: 0, dz: 0, mag: 0 }, turn: 0, snap: { armed: true }, headYaw: 0 };
  if (!xr || !xr.supported) {
    onStatus?.({ available: false, reason: "this browser has no WebXR" });
    return state;
  }
  const rig = new Entity("xr-rig");
  app.root.addChild(rig);
  let button = null;
  const parentBefore = camera.parent;

  const show = (available) => {
    onStatus?.({ available, reason: available ? null : "no VR headset is offered to this page (immersive-vr unavailable)" });
    if (!available || button) return;
    button = document.createElement("button");
    button.textContent = "Enter VR";
    button.className = "xr-enter";
    Object.assign(button.style, { position: "absolute", right: "14px", bottom: "14px", zIndex: 30, padding: "10px 16px",
      borderRadius: "8px", border: "1px solid #30363d", background: "#1f6feb", color: "#fff", font: "600 14px system-ui", cursor: "pointer" });
    button.onclick = () => start();
    canvas.parentElement?.appendChild(button);
  };
  xr.on("available:" + XRTYPE_VR, show);
  show(xr.isAvailable(XRTYPE_VR));

  function start() {
    if (state.active) return;
    camera.reparent(rig);
    camera.setLocalPosition(0, 0, 0);
    camera.setLocalEulerAngles(0, 0, 0);
    camera.camera.startXr(XRTYPE_VR, XRSPACE_LOCALFLOOR, {
      callback: (error) => {
        if (error) { camera.reparent(parentBefore); onStatus?.({ available: true, reason: `VR session refused: ${error.message}` }); }
      },
    });
  }
  xr.on("start", () => { state.active = true; if (button) button.style.display = "none"; });
  xr.on("end", () => {
    state.active = false;
    camera.reparent(parentBefore);
    if (button) button.style.display = "";
  });
  xr.input.on("selectstart", (source) => { if (source.handedness !== "left") onTrigger?.(true); });
  xr.input.on("selectend", (source) => { if (source.handedness !== "left") onTrigger?.(false); });

  state.update = (feet, bodyYaw) => {
    if (!state.active) return;
    const fwd = camera.forward;
    state.headYaw = yawOf(fwd);
    let lx = 0, ly = 0, rx = 0;
    for (const source of xr.input.inputSources) {
      const pad = source.gamepad;
      if (!pad || pad.axes.length < 4) continue;
      if (source.handedness === "left") { lx = pad.axes[2]; ly = pad.axes[3]; }
      else if (source.handedness === "right") rx = pad.axes[2];
    }
    state.move = stickToMove(lx, ly, state.headYaw);
    state.turn = snapTurn(state.snap, rx);
    rig.setPosition(feet.x, feet.y, feet.z);
    rig.setLocalEulerAngles(0, (bodyYaw + state.turn) * 180 / Math.PI, 0);
  };
  state.aimRay = () => {
    if (!state.active) return null;
    const source = xr.input.inputSources.find((s) => s.handedness === "right" && s.targetRayMode === "tracked-pointer");
    if (!source) return null;
    const o = source.getOrigin(), d = source.getDirection();
    return { origin: { x: o.x, y: o.y, z: o.z }, dir: { x: d.x, y: d.y, z: d.z } };
  };
  state.end = () => { if (state.active) camera.camera.endXr(); };
  state.rig = rig;
  return state;
}

