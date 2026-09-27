import "./workspace-modes.css";

const names: Record<string, string> = {
  sofa: "Three-seat sofa", armchair: "Armchair", coffee_table: "Coffee table", dining_table: "Dining table",
  dining_chair: "Dining chair", bed_double: "Double bed", desk: "Desk", wardrobe: "Wardrobe",
  bookshelf: "Bookshelf", side_table: "Side table", tv_stand: "TV stand", stool: "Stool",
};

/** Opaque, normalized top-view symbols; Placement.size supplies the real dimensions. */
export function FurnitureSymbol({ item }: { item: string }) {
  let drawing;
  if (item === "sofa" || item === "armchair") {
    const count = item === "sofa" ? 3 : 1, seat = 76 / count;
    drawing = <>
      <rect className="furniture-symbol__body" x="1" y="1" width="98" height="98" rx="8" />
      <rect className="furniture-symbol__cushion" x="10" y="3" width="80" height="25" rx="5" />
      {Array.from({ length: count }, (_, i) => <rect key={i} className="furniture-symbol__cushion" x={12 + i * seat} y="31" width={seat - 2} height="61" rx="6" />)}
      <rect className="furniture-symbol__body" x="2" y="24" width="10" height="72" rx="4" />
      <rect className="furniture-symbol__body" x="88" y="24" width="10" height="72" rx="4" />
      <rect className="furniture-symbol__pillow" x="16" y="29" width="15" height="23" rx="4" transform="rotate(-9 23 40)" />
      <rect className="furniture-symbol__pillow" x="69" y="29" width="15" height="23" rx="4" transform="rotate(9 77 40)" />
    </>;
  } else if (item === "bed_double") {
    drawing = <>
      <rect className="furniture-symbol__body" x="1" y="1" width="98" height="98" rx="3" />
      <rect className="furniture-symbol__support" x="1" y="1" width="7" height="98" rx="2" />
      <rect className="furniture-symbol__cushion" x="10" y="5" width="86" height="90" rx="5" />
      {[12, 55].map((y) => <rect key={y} className="furniture-symbol__pillow" x="13" y={y} width="25" height="33" rx="5" />)}
      <rect className="furniture-symbol__body" x="42" y="5" width="54" height="90" rx="3" />
      <path className="furniture-symbol__line" d="M49 5V95M86 8V92" />
    </>;
  } else if (["coffee_table", "dining_table", "desk"].includes(item)) {
    drawing = <>
      {[8, 80].flatMap((x) => [1, 80].map((y) => <rect key={`${x}-${y}`} className="furniture-symbol__support" x={x} y={y} width="12" height="19" rx="2" />))}
      <rect className="furniture-symbol__wood" x="1" y="9" width="98" height="82" rx={item === "dining_table" ? 16 : 5} />
      <path className="furniture-symbol__line" d="M14 18H86M14 82H86" />
      {item === "desk" ? <><rect className="furniture-symbol__support" x="30" y="20" width="40" height="5" rx="1" /><path className="furniture-symbol__line" d="M50 25V43M42 43H58" /><rect className="furniture-symbol__pillow" x="29" y="53" width="42" height="16" rx="2" /><path className="furniture-symbol__line" d="M34 59H66M34 64H66" /></> :
        item === "coffee_table" ? <><rect className="furniture-symbol__pillow" x="56" y="33" width="24" height="28" rx="1" transform="rotate(10 68 47)" /><circle className="furniture-symbol__body" cx="29" cy="50" r="9" /></> : null}
    </>;
  } else if (item === "side_table" || item === "stool") {
    drawing = <>
      <path className="furniture-symbol__support" d="M12 8L92 88L88 92L8 12ZM88 8L8 88L12 92L92 12Z" />
      <circle className={item === "stool" ? "furniture-symbol__cushion" : "furniture-symbol__wood"} cx="50" cy="50" r="43" />
      <circle className="furniture-symbol__line" cx="50" cy="50" r="35" />
    </>;
  } else if (item === "dining_chair") {
    drawing = <>
      {[8, 81].flatMap((x) => [1, 83].map((y) => <rect key={`${x}-${y}`} className="furniture-symbol__support" x={x} y={y} width="11" height="16" rx="2" />))}
      <rect className="furniture-symbol__wood" x="5" y="1" width="90" height="22" rx="5" />
      <rect className="furniture-symbol__cushion" x="9" y="29" width="82" height="66" rx="10" />
      <path className="furniture-symbol__line" d="M16 39Q50 33 84 39" />
    </>;
  } else if (item === "bookshelf") {
    drawing = <>
      <rect className="furniture-symbol__wood" x="1" y="1" width="98" height="98" rx="2" />
      {[7, 37, 67].map((y) => <g key={y}>
        {[8, 21, 31, 47, 59, 75, 85].map((x, i) => <rect key={x} className={i % 2 ? "furniture-symbol__body" : "furniture-symbol__pillow"} x={x} y={y + (i % 3) * 2} width={i % 2 ? 7 : 10} height={22 - (i % 3) * 2} rx="1" />)}
        <path className="furniture-symbol__line" d={`M4 ${y + 25}H96`} />
      </g>)}
    </>;
  } else if (item === "wardrobe" || item === "tv_stand") {
    drawing = <>
      <rect className="furniture-symbol__wood" x="1" y="1" width="98" height="98" rx="3" />
      <path className="furniture-symbol__line" d="M50 4V96M5 90H95" />
      <rect className="furniture-symbol__support" x="41" y="46" width="3" height="16" rx="1" />
      <rect className="furniture-symbol__support" x="56" y="46" width="3" height="16" rx="1" />
      {item === "tv_stand" && <><rect className="furniture-symbol__support" x="10" y="14" width="80" height="9" rx="2" /><path className="furniture-symbol__line" d="M50 23V36M37 36H63" /></>}
    </>;
  } else {
    drawing = <><rect className="furniture-symbol__body" x="4" y="4" width="92" height="92" rx="8" /><path className="furniture-symbol__line" d="M30 38Q30 22 50 22Q72 22 70 39Q68 50 50 55V66M50 76V80" /></>;
  }
  return <svg className="furniture-symbol" viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label={`${names[item] ?? item.replaceAll("_", " ")} top view`} focusable="false">{drawing}</svg>;
}
