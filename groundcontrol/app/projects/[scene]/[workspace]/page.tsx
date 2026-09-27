import { notFound } from "next/navigation";
import { ProjectWorkspace } from "@/components/project-workspace";

export default async function WorkspacePage({ params }: { params: Promise<{ scene: string; workspace: string }> }) {
  const { scene, workspace } = await params;
  if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/.test(scene) || scene.includes("..") || !["measure", "place", "plan", "mission", "ops", "inspect", "twin", "walk"].includes(workspace)) notFound();
  return <ProjectWorkspace key={`${scene}-${workspace}`} scene={scene} initialTab={workspace} />;
}
