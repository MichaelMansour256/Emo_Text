"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useSession } from "@/components/Session";
import { StatusBadge } from "@/components/StatusBadge";
import {
  api,
  ApiError,
  formatDuration,
  formatWords,
  Options,
  plural,
  Progress,
  Project,
  RenderRequest,
  watchProject,
} from "@/lib/api";

const STAGES: Record<string, string> = {
  ingest: "Reading the book",
  direct: "Directing: speakers and emotions",
  render: "Recording lines",
  assemble: "Building the audiobook",
};

export default function ProjectPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const { me, loading: sessionLoading, refresh } = useSession();
  const [project, setProject] = useState<Project | null>(null);
  const [options, setOptions] = useState<Options | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      setProject(await api.project(id));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  }, [id]);

  useEffect(() => {
    load();
    api.options().then(setOptions, () => setError("Can't load render options."));
  }, [load]);

  const active = project?.status === "queued" || project?.status === "rendering";

  // Live progress over server-sent events, with a slow poll as a safety net.
  useEffect(() => {
    if (!active) return;
    const stop = watchProject(id, (e) => {
      setProject((p) => (p ? { ...p, status: e.status, progress: e.progress, error: e.error } : p));
      if (e.status !== "queued" && e.status !== "rendering") {
        load();
        refresh();
      }
    });
    const poll = setInterval(load, 5000);
    return () => {
      stop();
      clearInterval(poll);
    };
  }, [active, id, load, refresh]);

  async function startRender(req: RenderRequest) {
    setBusy(true);
    setError(null);
    try {
      setProject(await api.render(id, req));
      refresh();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function remove() {
    if (!confirm(`Delete “${project?.title}” and its audio? This can't be undone.`)) return;
    await api.remove(id);
    router.push("/");
  }

  if (sessionLoading) return <p className="muted">Loading…</p>;
  if (!me) return <p className="muted">Sign in to see this book.</p>;
  if (!project) return error ? <p className="error">{error}</p> : <p className="muted">Loading…</p>;

  return (
    <>
      <Link href="/" className="small muted back">
        ← All books
      </Link>
      <div className="title-row">
        <div>
          <h1>{project.title}</h1>
          <p className="muted">
            {project.author ? `${project.author} · ` : ""}
            {plural(project.chapters.length, "chapter")} · {plural(project.words, "word")} ·{" "}
            {project.filename}
          </p>
        </div>
        <StatusBadge status={project.status} />
      </div>

      {error && <p className="error">{error}</p>}

      {active && <ProgressPanel progress={project.progress} status={project.status} />}
      {project.status === "failed" && (
        <div className="card error-card">
          <strong>The render failed.</strong> Your credits were refunded.
          {project.error && <pre className="small">{project.error}</pre>}
        </div>
      )}
      {!active && project.result && <Player project={project} />}

      {!active && options && (
        <RenderPanel
          project={project}
          options={options}
          credits={me.credits}
          busy={busy}
          onRender={startRender}
        />
      )}

      <h2>Chapters</h2>
      <ol className="chapters">
        {project.chapters.map((c) => (
          <li key={c.index} className="card">
            <div className="chapter-head">
              <strong>{c.title}</strong>
              <span className="muted small">{formatWords(c.words)} words</span>
            </div>
            <p className="small muted clamp">{c.preview}</p>
          </li>
        ))}
      </ol>

      <button className="danger link" onClick={remove}>
        Delete this book
      </button>
    </>
  );
}

function ProgressPanel({ progress, status }: { progress: Progress; status: string }) {
  const pct = progress && progress.total ? Math.round((100 * progress.done) / progress.total) : 0;
  const label =
    status === "queued" ? "Waiting for a free worker" : STAGES[progress?.stage ?? ""] ?? "Starting";
  return (
    <div className="card progress-card" aria-live="polite">
      <div className="chapter-head">
        <strong>{label}…</strong>
        {progress?.stage === "render" && (
          <span className="muted small">
            {progress.done} / {progress.total} lines
          </span>
        )}
      </div>
      <div className="bar" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div style={{ width: `${progress?.stage === "render" ? pct : 4}%` }} />
      </div>
    </div>
  );
}

function RenderPanel({
  project,
  options,
  credits,
  busy,
  onRender,
}: {
  project: Project;
  options: Options;
  credits: number;
  busy: boolean;
  onRender: (req: RenderRequest) => void;
}) {
  const available = options.providers.filter((p) => p.available);
  const prev = project.render_options as RenderRequest;
  const [provider, setProvider] = useState(
    prev.provider ?? (available.some((p) => p.name === options.default_provider)
      ? options.default_provider
      : available[0]?.name),
  );
  const engine = available.find((p) => p.name === provider);
  const [voice, setVoice] = useState(prev.voice ?? "");
  const [director, setDirector] = useState(prev.director ?? options.default_director);
  const [chapters, setChapters] = useState(project.chapters.length);
  const [qa, setQa] = useState(Boolean(prev.qa));

  const cost = useMemo(
    () => project.chapters.slice(0, chapters).reduce((sum, c) => sum + c.words, 0),
    [project.chapters, chapters],
  );
  const base: RenderRequest = {
    provider,
    voice: voice || undefined,
    director,
    max_chapters: chapters < project.chapters.length ? chapters : undefined,
    qa: options.qa ? qa : false,
  };

  return (
    <section className="card render-panel">
      <h2>{project.status === "draft" ? "Make the audiobook" : "Render again"}</h2>
      <div className="grid">
        <label>
          Voice engine
          <select value={provider} onChange={(e) => (setProvider(e.target.value), setVoice(""))}>
            {options.providers.map((p) => (
              <option key={p.name} value={p.name} disabled={!p.available}>
                {p.name}
                {p.emotion ? " (expressive)" : ""}
                {p.available ? "" : " — not configured"}
              </option>
            ))}
          </select>
        </label>
        <label>
          Narrator voice
          <select value={voice} onChange={(e) => setVoice(e.target.value)}>
            <option value="">Default ({engine?.default_voice})</option>
            {engine?.voices.map((v) => (
              <option key={v.id} value={v.id}>
                {v.label ?? v.id}
                {v.gender !== "unknown" ? ` · ${v.gender}` : ""}
              </option>
            ))}
          </select>
        </label>
        <label>
          Director
          <select value={director} onChange={(e) => setDirector(e.target.value)}>
            {options.directors.map((d) => (
              <option key={d} value={d}>
                {d === "none" ? "None (single narrator)" : d === "claude" ? "Claude (AI)" : "Built-in"}
              </option>
            ))}
          </select>
        </label>
        <label>
          Chapters
          <select value={chapters} onChange={(e) => setChapters(Number(e.target.value))}>
            {project.chapters.map((c) => (
              <option key={c.index} value={c.index}>
                {c.index === project.chapters.length ? `All ${c.index}` : `First ${c.index}`}
              </option>
            ))}
          </select>
        </label>
      </div>
      {options.qa && (
        <label className="inline-check small">
          <input type="checkbox" checked={qa} onChange={(e) => setQa(e.target.checked)} />
          Listening check: transcribe every line and flag the ones that don&apos;t match the text
        </label>
      )}
      {project.pending_edits > 0 && (
        <p className="small error">
          A new render reads and directs the book again, replacing your{" "}
          {plural(project.pending_edits, "line edit")}. To keep them, use{" "}
          <Link href={`/projects/${project.id}/edit`}>Edit lines → Update audio</Link>.
        </p>
      )}
      <div className="actions">
        <button disabled={busy || !engine} onClick={() => onRender({ ...base, preview: true })}>
          Free preview · first ~{options.preview_words} words
        </button>
        <button
          className="primary"
          disabled={busy || !engine || cost > credits}
          onClick={() => onRender(base)}
        >
          Render {chapters === project.chapters.length ? "the book" : `${chapters} chapters`} ·{" "}
          {formatWords(cost)} credits
        </button>
      </div>
      {cost > credits && (
        <p className="small">
          You have {formatWords(credits)} credits. <Link href="/billing">Buy more</Link> or render
          fewer chapters.
        </p>
      )}
    </section>
  );
}

function Player({ project }: { project: Project }) {
  const result = project.result!;
  const cast = Object.entries(result.cast.characters ?? {});
  return (
    <section className="card player">
      <div className="chapter-head">
        <h2>{result.preview ? "Preview" : "Listen"}</h2>
        <span className="muted small">
          {formatDuration(result.duration_s)} · {result.provider} voices · {result.director} director
        </span>
      </div>
      {result.preview && (
        <p className="small muted">The opening of chapter 1. Render the book to hear it all.</p>
      )}
      {project.status === "failed" && (
        <p className="small muted">This is the audio from before the failed update.</p>
      )}
      <div className="actions tight">
        <Link className="button" href={`/projects/${project.id}/edit`}>
          Edit lines and pronunciations
          {project.pending_edits ? ` (${project.pending_edits} waiting)` : ""}
        </Link>
        {result.qa && (
          <span className={`small ${result.qa.flagged ? "error" : "ok"}`}>
            Listening check: {result.qa.flagged ? `${plural(result.qa.flagged, "line")} to review` : "every line matched"}{" "}
            ({result.qa.checked} checked)
          </span>
        )}
      </div>
      <ol className="tracks">
        {result.chapters.map((c) => (
          <li key={c.file}>
            <div className="chapter-head">
              <span>{c.title}</span>
              <span className="muted small">
                {formatDuration(c.duration_s)} ·{" "}
                <a href={api.fileUrl(c.url)} download>
                  MP3
                </a>
              </span>
            </div>
            <audio controls preload="none" src={api.fileUrl(c.url)} />
          </li>
        ))}
      </ol>
      {result.m4b_url && (
        <a className="button primary" href={api.fileUrl(result.m4b_url)} download>
          Download audiobook (M4B with chapters)
        </a>
      )}
      {cast.length > 0 && (
        <>
          <h3>Cast</h3>
          <table className="cast">
            <tbody>
              <tr>
                <td>Narrator</td>
                <td className="muted">{result.cast.narrator}</td>
              </tr>
              {cast.map(([name, voice]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td className="muted">{voice}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}
