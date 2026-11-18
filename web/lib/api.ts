// Typed client for the Emo_Text backend.
//
// Auth: in dev mode the backend trusts an X-Dev-User email header, kept in localStorage.
// In production, call setTokenGetter() with your auth provider's token function
// (Clerk: useAuth().getToken, Supabase: session.access_token) and run the backend with
// EMOTEXT_AUTH_MODE=jwt.

export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "/api";

export type Voice = { id: string; gender: string; label: string | null };
export type Provider = {
  name: string;
  available: boolean;
  reason?: string;
  default_voice?: string;
  emotion?: boolean;
  voices: Voice[];
};
export type Options = {
  providers: Provider[];
  default_provider: string;
  directors: string[];
  default_director: string;
  preview_words: number;
  packs: { id: string; credits: number; price_cents: number }[];
  payments: boolean;
  max_upload_mb: number;
};
export type Me = { id: string; email: string | null; credits: number; auth_mode: string };
export type ChapterInfo = { index: number; title: string; words: number; preview: string };
export type Progress = { stage: string; done: number; total: number } | null;
export type RenderedChapter = {
  title: string;
  file: string;
  url: string;
  lines: number;
  words: number;
  duration_s: number;
};
export type RenderResult = {
  title: string;
  provider: string;
  director: string;
  preview: boolean;
  duration_s: number;
  chapters: RenderedChapter[];
  m4b: string | null;
  m4b_url?: string;
  characters: { name: string; gender: string; description?: string }[];
  cast: { narrator?: string; characters?: Record<string, string> };
};
export type ProjectSummary = {
  id: string;
  title: string;
  author: string | null;
  filename: string;
  words: number;
  chapter_count: number;
  status: "draft" | "queued" | "rendering" | "done" | "failed";
  error: string | null;
  created_at: string;
  updated_at: string;
};
export type Project = ProjectSummary & {
  chapters: ChapterInfo[];
  render_options: Record<string, unknown>;
  charged: number;
  progress: Progress;
  result: RenderResult | null;
};
export type RenderRequest = {
  provider?: string;
  voice?: string;
  director?: string;
  max_chapters?: number;
  preview?: boolean;
  cast?: Record<string, string>;
};

const DEV_USER_KEY = "emotext.devUser";
let tokenGetter: (() => Promise<string | null>) | null = null;

export function setTokenGetter(fn: (() => Promise<string | null>) | null) {
  tokenGetter = fn;
}

export function getDevUser(): string | null {
  try {
    return localStorage.getItem(DEV_USER_KEY);
  } catch {
    return null;
  }
}

export function setDevUser(email: string | null) {
  try {
    if (email) localStorage.setItem(DEV_USER_KEY, email);
    else localStorage.removeItem(DEV_USER_KEY);
  } catch {
    /* storage unavailable: the session just won't persist */
  }
}

async function authHeaders(): Promise<Record<string, string>> {
  if (tokenGetter) {
    const token = await tokenGetter();
    return token ? { Authorization: `Bearer ${token}` } : {};
  }
  const dev = getDevUser();
  return dev ? { "X-Dev-User": dev } : {};
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = { ...(await authHeaders()), ...(init.headers as Record<string, string>) };
  const resp = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (!resp.ok) {
    let message = resp.statusText;
    try {
      const body = await resp.json();
      message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(resp.status, message);
  }
  return resp.status === 204 ? (undefined as T) : resp.json();
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  me: () => request<Me>("/v1/me"),
  options: () => request<Options>("/v1/options"),
  projects: () => request<ProjectSummary[]>("/v1/projects"),
  project: (id: string) => request<Project>(`/v1/projects/${id}`),
  upload: (file: File, rightsConfirmed: boolean) => {
    const form = new FormData();
    form.append("file", file);
    form.append("rights_confirmed", String(rightsConfirmed));
    return request<Project>("/v1/projects", { method: "POST", body: form });
  },
  render: (id: string, req: RenderRequest) =>
    request<Project>(`/v1/projects/${id}/render`, json(req)),
  remove: (id: string) => request<void>(`/v1/projects/${id}`, { method: "DELETE" }),
  checkout: (pack: string) => request<{ url: string }>("/v1/billing/checkout", json({ pack })),
  ledger: () =>
    request<{ delta: number; reason: string; project_id: string | null; created_at: string }[]>(
      "/v1/billing/ledger",
    ),
  // Bucket storage returns absolute presigned URLs; local storage returns API paths.
  fileUrl: (path: string) => (/^https?:\/\//.test(path) ? path : `${API_BASE}${path}`),
};

export type ProjectEvent = { status: Project["status"]; progress: Progress; error: string | null };

/** Follow a render over server-sent events. fetch() instead of EventSource so the auth
 *  header can be sent. Returns a function that stops listening. */
export function watchProject(id: string, onEvent: (e: ProjectEvent) => void): () => void {
  const controller = new AbortController();
  (async () => {
    const resp = await fetch(`${API_BASE}/v1/projects/${id}/events`, {
      headers: await authHeaders(),
      signal: controller.signal,
    });
    if (!resp.ok || !resp.body) return;
    const reader = resp.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return;
      buffer += value;
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        const data = chunk
          .split("\n")
          .filter((l) => l.startsWith("data: "))
          .map((l) => l.slice(6))
          .join("\n");
        if (data) onEvent(JSON.parse(data));
      }
    }
  })().catch(() => {
    /* aborted or connection dropped; the page falls back to polling */
  });
  return () => controller.abort();
}

export const formatWords = (n: number) => n.toLocaleString("en-US");
export const formatDuration = (s: number) => {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.round(s % 60);
  return h ? `${h}h ${m}m` : `${m}:${String(sec).padStart(2, "0")}`;
};
export const plural = (n: number, word: string) =>
  `${n.toLocaleString("en-US")} ${word}${n === 1 ? "" : "s"}`;
