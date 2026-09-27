/* Pure VR input maths (no engine): thumbsticks to movement, snap turn, head yaw.
   Kept apart from vr.js so it can be tested in node without PlayCanvas. */
export const DEADZONE = 0.15;
export const SNAP_DEG = 30;
const SNAP_ARM = 0.3, SNAP_FIRE = 0.7;

/** Thumbstick (x right, y forward-negative as gamepads report) to a world move direction. */
export function stickToMove(x, y, headYaw, deadzone = DEADZONE) {
  const mag = Math.hypot(x, y);
  if (mag < deadzone) return { dx: 0, dz: 0, mag: 0 };
  const scale = Math.min(1, (mag - deadzone) / (1 - deadzone)) / mag;
  const mx = x * scale, mz = -y * scale;                  // forward is -y on a gamepad
  const s = Math.sin(headYaw), c = Math.cos(headYaw);
  // Same convention as the desktop walk: yaw 0 looks down -z.
  return { dx: mz * -s + mx * c, dz: mz * -c + mx * -s, mag: Math.hypot(mx, mz) };
}

/** Snap turn with hysteresis: one turn per push, re-armed when the stick returns. */
export function snapTurn(state, axisX) {
  if (!state.armed) {
    if (Math.abs(axisX) < SNAP_ARM) state.armed = true;
    return 0;
  }
  if (Math.abs(axisX) > SNAP_FIRE) {
    state.armed = false;
    return -Math.sign(axisX) * (SNAP_DEG * Math.PI) / 180;   // stick right turns right (yaw decreases)
  }
  return 0;
}

/** Yaw of a camera's forward vector, in the walk convention (0 = looking down -z). */
export function yawOf(forward) {
  return Math.atan2(-forward.x, -forward.z);
}
