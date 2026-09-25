import { redirect } from "next/navigation";

export default async function Deliverables({ searchParams }: { searchParams: Promise<{ scene?: string }> }) {
  const { scene } = await searchParams;
  redirect(scene ? `/projects/${encodeURIComponent(scene)}?tab=exports` : "/");
}
