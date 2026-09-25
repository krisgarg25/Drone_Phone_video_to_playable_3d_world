import { redirect } from "next/navigation";

export default async function Survey({ searchParams }: { searchParams: Promise<{ scene?: string }> }) {
  const { scene } = await searchParams;
  redirect(scene ? `/projects/${encodeURIComponent(scene)}?tab=details` : "/");
}
