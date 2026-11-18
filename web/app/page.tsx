"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { useSession } from "@/components/Session";
import { StatusBadge } from "@/components/StatusBadge";
import { api, ApiError, plural, ProjectSummary } from "@/lib/api";

const ACCEPT = ".pdf,.epub,.txt,.md";

export default function Home() {
  const { me, loading, error } = useSession();
  if (loading) return <p className="muted">Loading…</p>;
  return me ? <Library /> : <Welcome error={error} />;
}

function Welcome({ error }: { error: string | null }) {
  const { signIn, mode } = useSession();
  const [email, setEmail] = useState("");
  return (
    <section className="hero">
      <h1>Your book, performed.</h1>
      <p className="lede">
        Upload a PDF or EPUB. Emo_Text finds the chapters, works out who is speaking, gives every
        character their own voice and reads each line with the right emotion.
      </p>
      {mode === "clerk" ? (
        <div className="signin">
          <button className="primary" onClick={() => signIn("")}>
            Sign in or create an account
          </button>
          <p className="small muted">New accounts get free credits to try it.</p>
          {error && <p className="error">{error}</p>}
        </div>
      ) : (
      <form
        className="signin card"
        onSubmit={(e) => {
          e.preventDefault();
          if (email.includes("@")) signIn(email);
        }}
      >
        <label htmlFor="email">Email</label>
        <div className="row">
          <input
            id="email"
            type="email"
            required
            placeholder="you@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
          <button className="primary" type="submit">
            Get started
          </button>
        </div>
        <p className="small muted">
          Development sign-in: no password. New accounts get free credits to try it.
        </p>
        {error && <p className="error">{error}</p>}
      </form>
      )}
      <ul className="features">
        <li>
          <strong>AI Director</strong>
          <span>Speaker and emotion for every line of dialogue.</span>
        </li>
        <li>
          <strong>Full cast</strong>
          <span>A consistent voice per character, chosen by gender.</span>
        </li>
        <li>
          <strong>Chaptered M4B</strong>
          <span>Ready for any audiobook player, plus MP3 per chapter.</span>
        </li>
      </ul>
    </section>
  );
}

function Library() {
  const router = useRouter();
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [rights, setRights] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    try {
      setProjects(await api.projects());
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function upload(file: File | undefined) {
    if (!file) return;
    if (!rights) {
      setError("Please confirm you have the rights to this book first.");
      return;
    }
    setUploading(true);
    setError(null);
    try {
      const project = await api.upload(file, rights);
      router.push(`/projects/${project.id}`);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
      setUploading(false);
    }
  }

  return (
    <>
      <label className="consent">
        <input type="checkbox" checked={rights} onChange={(e) => setRights(e.target.checked)} />
        <span>
          I own this book or have permission to turn it into audio, and I agree to the{" "}
          <Link href="/terms">Terms</Link>.
        </span>
      </label>
      <div
        className={`dropzone ${dragging ? "dragging" : ""} ${rights ? "" : "locked"}`}
        aria-disabled={!rights}
        role="button"
        tabIndex={0}
        onClick={() => input.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          upload(e.dataTransfer.files[0]);
        }}
      >
        <input
          ref={input}
          type="file"
          accept={ACCEPT}
          hidden
          onChange={(e) => upload(e.target.files?.[0])}
        />
        <strong>{uploading ? "Reading your book…" : "Drop a book here, or click to choose"}</strong>
        <span className="muted small">PDF, EPUB, TXT or Markdown</span>
      </div>
      {error && <p className="error">{error}</p>}

      <h2>Your books</h2>
      {projects === null ? (
        <p className="muted">Loading…</p>
      ) : projects.length === 0 ? (
        <p className="muted">No books yet. Upload one to see its chapters and hear a free preview.</p>
      ) : (
        <ul className="project-list">
          {projects.map((p) => (
            <li key={p.id}>
              <Link href={`/projects/${p.id}`} className="card project-row">
                <div>
                  <strong>{p.title}</strong>
                  <span className="muted small">
                    {p.author ? `${p.author} · ` : ""}
                    {plural(p.chapter_count, "chapter")} · {plural(p.words, "word")}
                  </span>
                </div>
                <StatusBadge status={p.status} />
              </Link>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}
