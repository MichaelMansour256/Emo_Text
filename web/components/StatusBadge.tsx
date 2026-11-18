import type { ProjectSummary } from "@/lib/api";

const LABELS: Record<ProjectSummary["status"], string> = {
  draft: "Not rendered",
  queued: "Queued",
  rendering: "Rendering",
  done: "Ready",
  failed: "Failed",
};

export function StatusBadge({ status }: { status: ProjectSummary["status"] }) {
  return <span className={`badge badge-${status}`}>{LABELS[status]}</span>;
}
