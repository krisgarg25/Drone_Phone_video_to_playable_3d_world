# Design: Ground Control film

**Concept.** The product is the proof. Every graphic is drawn in the app's own instrument language: graphite panels, Geist, and amber only where something is live. The film reads as one continuous piece of the software, not an ad wrapped around it.

## Palette (from `groundcontrol/app/studio.css`)

| Role | Value | Use |
|---|---|---|
| ground | `hsl(225 12% 5%)` | every background |
| panel | `hsl(222 12% 8%)` | cards, mini windows |
| raised | `hsl(222 11% 11%)` | window title bars |
| line | `hsl(222 9% 22%)` | 2 px borders, rules |
| ink | `hsl(220 25% 96%)` | headlines, body |
| dim | `hsl(218 12% 76%)` | secondary text |
| muted | `hsl(218 8% 57%)` | labels only (large sizes) |
| accent | `hsl(36 100% 57%)` | the one hue: live signal, numbers, the focal mark |
| green | `hsl(156 62% 52%)` | only "real" / "passed" |

Rules:
- One accent hue. Green appears only as the app's own pass state.
- No gradient text, no glows, no purple, no ALL-CAPS mono labels.

## Type

- **Geist** (variable, `assets/fonts/Geist-Variable.woff2`) is the one voice for statements: 300 against 700–800 for contrast, with -0.03em tracking on display sizes.
- **Geist Mono** is the data register: numbers, file names, step counters.
- Sizes: headlines 72–120 px, body 30–40 px, labels 22–26 px.

## Motion

- Default ease `power3.out` for entrances, with `expo.out` for the hero reveals. Entrances run 0.4–0.7 s.
- Transitions: the primary is a blur crossfade (calm to medium). The accent is a scale-back of the app footage into a feature board and out again. The outro fades to ground.
- Ambient: one slow drift on the ghost numerals of each board.

## Frame

- Edge-anchored: headlines top-left at a 96 px margin, and data anchored bottom or right.
- App footage is full bleed. Mini windows are graphite cards with a 44 px title bar, a 14 px radius and 2 px borders.
