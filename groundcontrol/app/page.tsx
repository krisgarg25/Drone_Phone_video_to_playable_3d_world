import { Suspense } from "react";
import { ProjectLibrary } from "@/components/project-library";

export default function Home() {
  return <Suspense fallback={<div className="studio loading-state">Opening workspace…</div>}><ProjectLibrary /></Suspense>;
}
