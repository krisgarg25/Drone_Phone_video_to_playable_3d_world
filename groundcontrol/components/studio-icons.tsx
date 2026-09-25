import type { SVGProps } from "react";

const paths = {
  grid: "M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z",
  plus: "M12 5v14 M5 12h14",
  search: "M21 21l-5-5 M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0",
  arrow: "M5 12h14 M13 6l6 6-6 6",
  back: "M19 12H5 M11 6l-6 6 6 6",
  chevron: "M9 5l7 7-7 7",
  cube: "M12 2l9 5v10l-9 5-9-5V7z M3 7l9 5 9-5 M12 12v10 M7 4.8l10 5.4",
  layers: "M12 3L2 8l10 5 10-5z M2 12l10 5 10-5 M2 16l10 5 10-5",
  upload: "M12 16V3 M7 8l5-5 5 5 M3 15v5a1 1 0 0 0 1 1h16a1 1 0 0 0 1-1v-5",
  download: "M12 3v13 M7 11l5 5 5-5 M3 17v4h18v-4",
  play: "M8 4l13 8-13 8z",
  clock: "M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0 M12 6v6l4 2",
  check: "M5 12l4 4L20 5",
  close: "M6 6l12 12 M6 18L18 6",
  settings: "M4 7h16 M4 17h16 M8 4v6 M16 14v6",
  image: "M3 3h18v18H3z M3 16l6-6 4 4 3-3 5 5 M16 7h.01",
  ruler: "M3 17L17 3l4 4L7 21z M7 13l2 2 M11 9l2 2 M15 5l2 2",
  pin: "M20 10c0 6-8 12-8 12S4 16 4 10a8 8 0 1 1 16 0 M15 10a3 3 0 1 1-6 0 3 3 0 0 1 6 0",
  globe: "M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0 M2 12h20 M12 2c6 6 6 14 0 20-6-6-6-14 0-20",
  activity: "M2 12h5l3-8 4 16 3-8h5",
  info: "M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0 M12 11v6 M12 7h.01",
  folder: "M3 5h6l2 3h10v12H3z",
  video: "M3 5h13v14H3z M16 10l5-3v10l-5-3",
  camera: "M3 6h5l2-3h4l2 3h5v14H3z M16 13a4 4 0 1 1-8 0 4 4 0 0 1 8 0",
  expand: "M3 9V3h6 M15 3h6v6 M21 15v6h-6 M9 21H3v-6",
  reset: "M3 10a9 9 0 1 1 2 9 M3 3v7h7",
  eye: "M2 12s4-7 10-7 10 7 10 7-4 7-10 7-10-7-10-7 M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0",
  stop: "M5 5h14v14H5z",
  trash: "M3 6h18 M9 6V3h6v3 M5 6l1 15h12l1-15 M10 10v7 M14 10v7",
  file: "M5 2h9l5 5v15H5z M14 2v6h5 M8 13h8 M8 17h6",
  link: "M10 14l4-4 M8 16l-1 1a4 4 0 0 1-6-6l4-4a4 4 0 0 1 6 0 M16 8l1-1a4 4 0 0 1 6 6l-4 4a4 4 0 0 1-6 0",
  help: "M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0 M9 8a3 3 0 1 1 5 2c-2 1-2 2-2 3 M12 17h.01",
  compass: "M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0 M16 8l-3 5-5 3 3-5z",
};
export type IconName = keyof typeof paths;
export function Icon({ name, size = 18, ...props }: SVGProps<SVGSVGElement> & { name: IconName; size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}><path d={paths[name]} /></svg>;
}
