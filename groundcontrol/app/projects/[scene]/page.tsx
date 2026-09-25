import { notFound } from "next/navigation";
import { ProjectWorkspace } from "@/components/project-workspace";

export default async function ProjectPage({ params, searchParams }: { params: Promise<{ scene: string }>; searchParams: Promise<{ tab?: string }> }) {
  const { scene } = await params;
  const { tab } = await searchParams;
  if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/.test(scene) || scene.includes("..")) notFound();
  const initialTab = tab && ["layers", "measure", "details", "exports"].includes(tab) ? tab : "layers";
  return <ProjectWorkspace key={`${scene}-${initialTab}`} scene={scene} initialTab={initialTab} />;
}
