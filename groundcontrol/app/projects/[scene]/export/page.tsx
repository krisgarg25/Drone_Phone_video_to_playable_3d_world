import { notFound } from "next/navigation";
import { ExportCenter } from "@/components/export-center";

export default async function ExportPage({ params }: { params: Promise<{ scene: string }> }) {
  const { scene } = await params;
  if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/.test(scene) || scene.includes("..")) notFound();
  return <ExportCenter scene={scene} />;
}
