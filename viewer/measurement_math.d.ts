export type MeasurementKind = 'point' | 'annotation' | 'distance' | 'height' | 'area' | 'volume' | null;
export type MeasurementUnit = 'm' | 'units';
export interface MeasurementPreview {
  value: number | null;
  unit: string;
  /** Euclidean edges; area/volume include the closing edge. */
  segments: number[];
  /** Accumulated XZ distance along the open path. */
  horizontal: number;
  /** Accumulated absolute Y changes along the open path. */
  vertical: number;
  /** Closed XZ footprint perimeter, never a volume. */
  perimeter: number;
}
/** Pure geometry estimate; never changes measurement validity or uncertainty. */
export function previewMeasurement(kind: MeasurementKind, points: ReadonlyArray<ReadonlyArray<number>>, unit?: MeasurementUnit): MeasurementPreview;
