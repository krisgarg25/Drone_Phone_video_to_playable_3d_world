# Voiceover script (v8, timed to the 7:02 cut)

Mostly English, spoken the way you'd explain your own project to someone you respect: calm, sincere, no hype. If a line sounds different in your own words, go with your words.

- **Bold** words get a little weight.
- `…` means pause and let the picture breathe.
- Measurements are approximate, so keep "around". Don't say the model "never saw" frame 41.
- Record each block as its own file (`vo/01.wav`, `vo/02.wav`…). The time is when you start speaking. If a line runs long, drop its last phrase rather than rushing.

---

## Part 1 · The test (0:00 – 0:18)

**01 · 0:00.6**
Let's start with something simple. One of these images is a real photo from a drone video. The other was rendered from a 3D model we built.

**02 · 0:04.4**
Take a moment… which one do you think is real?

*(stay quiet through the countdown)*

**03 · 0:08.3**
If you picked A, you're not alone. A is the render. B is the real photo.

**04 · 0:12.4**
How we made it comes later. First, let me show you what you can **do** with it.

---

## Part 2 · What it can do (0:18 – 5:40)

**05 · 0:18.6** (A 12-second drone flight)
All of this came from a single drone video, just **twelve seconds** long.

**06 · 0:23.9** (Explore)
This isn't a video playing back. It's real 3D geometry: the flight path, the collision mesh underneath, and every photo exactly where it was taken.

**07 · 0:37.1** (click a photo)
Click any photo, and it takes you to where the drone was looking from.

**08 · 0:40.9** (More in Explore)
It also understands what it's looking at: the ground, the rocks, the trees.

**09 · 0:57.0** (Measure: height)
Now let's measure. A click at the base, one at the top. Around **seven and a half metres**. No tape, no ladder.

**10 · 1:06.2** (More in Measure)
Distances, areas, the volume of a pile, and notes pinned right where you spotted something. Each one takes a couple of clicks.

**11 · 1:31.5** (Inspect)
This is one of my favourite parts. Pick any spot on the rock, and it finds **every photo** that captured it.

**12 · 1:43.3** (relief draped on the model)
For heritage sites, it brings out fine surface detail that the eye easily misses.

**13 · 1:53.9** (More in Inspect)
You get true-scale cross sections and a record of every file. And where there are poles or cables, it can measure tilt and sag, or show what changed since the last flight.

**14 · 2:16.0** (Operations: flood)
Now think about a flood. Click where the water comes in… and watch the valley fill, following the **real terrain**.

**15 · 2:28.4** (border line)
Draw a border, place a post, and see what it covers, and what it **doesn't**.

**16 · 2:34.5** (More in Operations)
Which roads are still usable, map tiles for GIS, even a relief camp laid out at real size.

**17 · 2:58.0** (the animated cards)
And with a second flight or a design model: damage grading, debris volume, cut and fill, and a check of what was built against the plan.

**18 · 3:08.6** (Twin)
A digital twin that shows three things: how the site looks, how it was measured, and how **confident** we are about each part.

**19 · 3:21.2** (Mission)
For mission planning, mark the enemy post, draw your route… and see which parts of it they can watch.

**20 · 3:45.7** (terrain)
Terrain shows where vehicles can move, before anyone has to go in.

**21 · 3:51.9** (More in Mission)
It can guess where they'd watch from, build a sand table for the briefing, rehearse at night, and find landing zones for a helicopter.

**22 · 4:15.2** (Walk)
And then… you can actually **walk through it**. The ground is solid, and the physics is real.

**23 · 4:27.1** (Export)
Pick the files you need, and take them away in one zip.

**24 · 4:37.2** (Open ground)
Here's a second site, open ground, for planning.

**25 · 4:42.4** (Plan)
Draw a building, add floors, lay a road past it… then swipe between what exists today and what you're proposing.

**26 · 5:07.6** (More in Plan)
Break the height limit, and the rules flag it right away. You see the impact of the scheme, and you can export straight to GIS or CAD.

**27 · 5:30.8** (Place)
And yes, you can place real-size furniture inside a real scan.

---

## Part 3 · How it's made (5:40 – 7:02)

**28 · 5:39.9** (the drone clip plays)
So how does a short drone video become all of this? It starts with just the video. No ground markers, no survey team.

**29 · 5:47.8** (files drop in and sort themselves)
If you have more, like a flight log, GPS or camera calibration, drop it in with the video. The system **works out what each file is** and uses it in the right place. And if a log doesn't state its accuracy, it tells you. It **never makes up a number**.

**30 · 6:01.5** (Reconstruct)
One button, and everything runs on this laptop. Nothing gets uploaded.

**31 · 6:10.4** (the steps light up)
There's no fixed recipe. The system **studies the video** first, then picks only the steps that video needs, and writes down why. For this flight, **twenty-four** of twenty-six.

**32 · 6:19.8** (keyframes)
It keeps only the sharpest frames.

**33 · 6:25.2** (camera solve)
Then it finds the same points across many photos, and places every camera in 3D.

**34 · 6:35.8** (training)
It trains on the GPU, and the model slowly learns to look just like the photos.

**35 · 6:44.5** (physics)
Underneath, a solid surface you can measure on and walk on.

**36 · 6:49.8** (render against the real photo)
Finally, it checks itself against the real photo. Remember that frame from the beginning? **This is where it came from.**

**37 · 6:56.2** (full screen, then black)
Twelve seconds of video, one laptop… and a world you can stand in.

*(silence till black)*

---

## Bonus · 60-second live intro (say this in the room before you press play)

Good morning, sir, ma'am. Today, if you want a site surveyed, it usually takes a day in the field and another three to five days before the results come back. But when there's a flood, or something happens at the border, nobody has that kind of time.

So we asked a simple question. What if an ordinary drone video, just twelve seconds long, could become a 3D world you can measure, plan on and walk through? On one laptop, without sending your data to the cloud.

Our system looks at the video and decides for itself which steps it needs. And wherever we aren't sure about something, we don't guess. We say so.

Rather than just talk about it, let me show you.

*(press play)*
