"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useSession } from "@/components/Session";
import {
  api,
  ApiError,
  EMOTIONS,
  formatWords,
  LineEdit,
  Options,
  ScriptLine,
  ScriptView,
} from "@/lib/api";

const PAGE = 200;
type Filter = "all" | "dialogue" | "flagged" | "edited";

const keyOf = (chapter: number, line: ScriptLine) => `${chapter}:${line.index}`;

export default function EditPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const { me, loading: sessionLoading, refresh } = useSession();
  const [view, setView] = useState<ScriptView | null>(null);
  const [options, setOptions] = useState<Options | null>(null);
  const [changes, setChanges] = useState<Record<string, LineEdit>>({});
  const [chapter, setChapter] = useState(0);
  const [filter, setFilter] = useState<Filter>("all");
  const [limit, setLimit] = useState(PAGE);
  const [qa, setQa] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const v = await api.script(id);
      setView(v);
      setQa(Boolean(v.qa));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  }, [id]);

  useEffect(() => {
    load();
    api.options().then(setOptions, () => undefined);
  }, [load]);

  const dirty = Object.keys(changes).length > 0;

  // Warn before leaving with unsaved changes.
  useEffect(() => {
    if (!dirty) return;
    const warn = (e: BeforeUnloadEvent) => e.preventDefault();
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  const speakers = useMemo(() => {
    const names = new Set<string>(["narrator", "unknown"]);
    view?.characters.forEach((c) => names.add(c.name));
    view?.chapters.forEach((ch) => ch.lines.forEach((ln) => names.add(ln.speaker)));
    return [...names];
  }, [view]);

  const lines = useMemo(() => {
    const all = view?.chapters[chapter]?.lines ?? [];
    return all.filter((ln) => {
      if (ln.kind === "title") return false;
      const changed = Boolean(changes[keyOf(chapter, ln)]);
      if (filter === "dialogue") return ln.kind === "dialogue";
      if (filter === "flagged") return ln.qa?.flagged;
      if (filter === "edited") return ln.edited || changed;
      return true;
    });
  }, [view, chapter, filter, changes]);

  const flaggedCount = useMemo(
    () =>
      view?.chapters.reduce((n, ch) => n + ch.lines.filter((ln) => ln.qa?.flagged).length, 0) ??
      0,
    [view],
  );

  function change(chapterIndex: number, line: ScriptLine, edit: LineEdit) {
    const key = keyOf(chapterIndex, line);
    setNotice(null);
    setChanges((prev) => {
      const next = { ...(prev[key] ?? {}), ...edit };
      // Drop fields that are back to the shown value.
      for (const field of Object.keys(next) as (keyof LineEdit)[]) {
        if (next[field] === line[field]) delete next[field];
      }
      const out = { ...prev };
      if (Object.keys(next).length) out[key] = next;
      else delete out[key];
      return out;
    });
  }

  /** Every pending edit: lines the server already marks as edited, plus local changes. */
  function allEdits(): Record<string, LineEdit> {
    const edits: Record<string, LineEdit> = {};
    view?.chapters.forEach((ch, c) =>
      ch.lines.forEach((ln) => {
        const key = keyOf(c, ln);
        if (ln.edited || changes[key]) {
          edits[key] = {
            speaker: ln.speaker,
            emotion: ln.emotion,
            intensity: ln.intensity,
            text: ln.text,
            ...changes[key],
          };
        }
      }),
    );
    return edits;
  }

  async function save(): Promise<ScriptView | null> {
    setBusy(true);
    setError(null);
    try {
      const v = await api.saveEdits(id, allEdits());
      setView(v);
      setChanges({});
      setNotice("Saved. Your edits apply the next time you update the audio.");
      return v;
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
      return null;
    } finally {
      setBusy(false);
    }
  }

  async function update() {
    const saved = dirty ? await save() : view;
    if (!saved || saved.pending_cost === 0) return;
    setBusy(true);
    try {
      await api.rerender(id, options?.qa ? qa : false);
      refresh();
      router.push(`/projects/${id}`);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
      setBusy(false);
    }
  }

  if (sessionLoading) return <p className="muted">Loading…</p>;
  if (!me) return <p className="muted">Sign in to edit this book.</p>;
  if (!view) return error ? <p className="error">{error}</p> : <p className="muted">Loading…</p>;

  const pendingCost = view.pending_cost;
  return (
    <>
      <Link href={`/projects/${id}`} className="small muted back">
        ← Back to the book
      </Link>
      <h1>Edit lines</h1>
      <p className="muted">
        Fix who says a line, how it is said, or its words. Only the lines you change are recorded
        again, and you pay only for their words.
      </p>

      <div className="card editor-bar">
        <div>
          <strong>
            {dirty
              ? `${Object.keys(changes).length} unsaved change${Object.keys(changes).length === 1 ? "" : "s"}`
              : view.pending_edits
                ? `${view.pending_edits} edited line${view.pending_edits === 1 ? "" : "s"} waiting`
                : "No changes yet"}
          </strong>
          {!dirty && pendingCost > 0 && (
            <span className="muted small"> · updating costs {formatWords(pendingCost)} credits</span>
          )}
        </div>
        <div className="actions tight">
          {options?.qa && (
            <label className="inline-check small">
              <input type="checkbox" checked={qa} onChange={(e) => setQa(e.target.checked)} />
              Listening check
            </label>
          )}
          <button disabled={busy || !dirty} onClick={save}>
            Save
          </button>
          <button
            className="primary"
            disabled={busy || (!dirty && pendingCost === 0) || pendingCost > me.credits}
            onClick={update}
          >
            Update audio
          </button>
        </div>
      </div>
      {error && <p className="error">{error}</p>}
      {notice && <p className="ok small">{notice}</p>}

      <Pronunciations
        id={id}
        initial={view.pronunciations}
        onSaved={(cost) => setView((v) => (v ? { ...v, pending_cost: cost } : v))}
      />

      <div className="editor-filters">
        <label>
          Chapter
          <select
            value={chapter}
            onChange={(e) => {
              setChapter(Number(e.target.value));
              setLimit(PAGE);
            }}
          >
            {view.chapters.map((ch, i) => (
              <option key={i} value={i}>
                {ch.chapter}
              </option>
            ))}
          </select>
        </label>
        <label>
          Show
          <select value={filter} onChange={(e) => setFilter(e.target.value as Filter)}>
            <option value="all">All lines</option>
            <option value="dialogue">Dialogue only</option>
            <option value="edited">Edited lines</option>
            {view.qa && <option value="flagged">Flagged by the listening check ({flaggedCount})</option>}
          </select>
        </label>
      </div>

      <datalist id="speakers">
        {speakers.map((s) => (
          <option key={s} value={s} />
        ))}
      </datalist>

      {lines.length === 0 && <p className="muted">No lines match.</p>}
      <ol className="lines">
        {lines.slice(0, limit).map((ln) => {
          const pending = changes[keyOf(chapter, ln)] ?? {};
          const shown = { ...ln, ...pending };
          const changed = ln.edited || Object.keys(pending).length > 0;
          return (
            <li key={ln.index} className={`card line ${changed ? "changed" : ""} ${ln.qa?.flagged ? "flagged" : ""}`}>
              <textarea
                aria-label="Line text"
                value={shown.text}
                rows={Math.min(6, Math.max(1, Math.ceil(shown.text.length / 90)))}
                onChange={(e) => change(chapter, ln, { text: e.target.value })}
              />
              <div className="line-controls">
                <label>
                  Speaker
                  <input
                    list="speakers"
                    value={shown.speaker}
                    onChange={(e) => change(chapter, ln, { speaker: e.target.value })}
                  />
                </label>
                <label>
                  Emotion
                  <select
                    value={shown.emotion}
                    onChange={(e) => change(chapter, ln, { emotion: e.target.value })}
                  >
                    {EMOTIONS.map((em) => (
                      <option key={em}>{em}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Strength {Math.round(shown.intensity * 100)}%
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.1}
                    value={shown.intensity}
                    onChange={(e) => change(chapter, ln, { intensity: Number(e.target.value) })}
                  />
                </label>
              </div>
              {ln.qa?.flagged && (
                <p className="small qa-note">
                  The listening check heard: <q>{ln.qa.heard || "(nothing)"}</q>
                </p>
              )}
              {changed && <span className="small muted">Edited</span>}
            </li>
          );
        })}
      </ol>
      {lines.length > limit && (
        <button onClick={() => setLimit((n) => n + PAGE)}>
          Show more ({lines.length - limit} left)
        </button>
      )}
    </>
  );
}

function Pronunciations({
  id,
  initial,
  onSaved,
}: {
  id: string;
  initial: Record<string, string>;
  onSaved: (pendingCost: number) => void;
}) {
  const [rows, setRows] = useState<[string, string][]>(() => Object.entries(initial));
  const [open, setOpen] = useState(Object.keys(initial).length > 0);
  const [state, setState] = useState<string | null>(null);

  async function save() {
    setState("Saving…");
    try {
      const out = await api.savePronunciations(id, Object.fromEntries(rows.filter(([t, s]) => t.trim() && s.trim())));
      setRows(Object.entries(out.pronunciations));
      onSaved(out.pending_cost);
      setState("Saved. They apply the next time you update the audio.");
    } catch (e) {
      setState(e instanceof ApiError ? e.message : String(e));
    }
  }

  return (
    <section className="card pronunciations">
      <button className="link" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        {open ? "▾" : "▸"} Pronunciations ({rows.length})
      </button>
      {open && (
        <>
          <p className="small muted">
            Spell names and words the way they should sound, e.g. Nadia → NAH-dee-ah. The book&apos;s
            text doesn&apos;t change; only what the voice reads.
          </p>
          {rows.map(([term, say], i) => (
            <div className="row pron-row" key={i}>
              <input
                aria-label="Word in the book"
                placeholder="Word in the book"
                value={term}
                onChange={(e) => setRows((r) => r.map((x, j) => (j === i ? [e.target.value, x[1]] : x)))}
              />
              <input
                aria-label="Say it as"
                placeholder="Say it as"
                value={say}
                onChange={(e) => setRows((r) => r.map((x, j) => (j === i ? [x[0], e.target.value] : x)))}
              />
              <button aria-label={`Remove ${term || "entry"}`} onClick={() => setRows((r) => r.filter((_, j) => j !== i))}>
                ✕
              </button>
            </div>
          ))}
          <div className="actions tight">
            <button onClick={() => setRows((r) => [...r, ["", ""]])}>Add a word</button>
            <button className="primary" onClick={save}>
              Save pronunciations
            </button>
          </div>
          {state && <p className="small muted">{state}</p>}
        </>
      )}
    </section>
  );
}
