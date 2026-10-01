import type { SVGProps } from "react";
import {
  AirplaneTilt, ArrowCounterClockwise, ArrowLeft, ArrowRight, ArrowUpRight, ArrowsOutCardinal, Bank, Buildings, Camera, CaretDown, CaretRight,
  Check, Clock, Compass, Crane, Crosshair, Cube, CubeTransparent, CornersOut, CursorClick, DownloadSimple, Drone, Eye, FileText, Flag, Folder,
  Globe, GridFour, Image, Info, Keyboard, Lightbulb, Link, MagnifyingGlass, MapPin, Mountains, Path, PersonSimpleWalk, Play, Plus, Pulse,
  Question, Ruler, Shield, SidebarSimple, SlidersHorizontal, Sparkle, SquaresFour, Stack, Stop, Target, Trash, UploadSimple, VideoCamera,
  Warning, Wrench, X, Angle, WaveSine, Scan, ClockCounterClockwise, Scroll, Images, ListChecks, SealCheck, MagnifyingGlassPlus, CaretLeft, Knife,
} from "@phosphor-icons/react/dist/ssr";

/** One icon set (Phosphor) behind the studio's own names, so call sites never import a library directly. */
const icons = {
  grid: GridFour, plus: Plus, search: MagnifyingGlass, arrow: ArrowRight, back: ArrowLeft, chevron: CaretRight, down: CaretDown,
  cube: Cube, layers: Stack, upload: UploadSimple, download: DownloadSimple, play: Play, clock: Clock, check: Check, close: X,
  settings: SlidersHorizontal, image: Image, ruler: Ruler, pin: MapPin, globe: Globe, activity: Pulse, info: Info, folder: Folder,
  video: VideoCamera, camera: Camera, expand: CornersOut, reset: ArrowCounterClockwise, eye: Eye, stop: Stop, trash: Trash, file: FileText,
  link: Link, help: Question, compass: Compass, apps: SquaresFour, target: Target, flag: Flag, shield: Shield, building: Buildings,
  alert: Warning, crane: Crane, mountain: Mountains, wrench: Wrench, columns: Bank, twin: CubeTransparent, radar: Crosshair, route: Path,
  sparkle: Sparkle, arrowUpRight: ArrowUpRight, cursor: CursorClick, move: ArrowsOutCardinal, fly: AirplaneTilt, walk: PersonSimpleWalk,
  keyboard: Keyboard, sidebar: SidebarSimple, tip: Lightbulb, drone: Drone,
  angle: Angle, wave: WaveSine, scan: Scan, history: ClockCounterClockwise, scroll: Scroll, photos: Images, list: ListChecks,
  seal: SealCheck, zoom: MagnifyingGlassPlus, left: CaretLeft, cut: Knife,
};
export type IconName = keyof typeof icons;
export function Icon({ name, size = 18, ...props }: SVGProps<SVGSVGElement> & { name: IconName; size?: number }) {
  const Glyph = icons[name];
  return <Glyph size={size} aria-hidden="true" {...(props as Record<string, unknown>)} />;
}
