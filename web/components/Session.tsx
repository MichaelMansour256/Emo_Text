"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { api, ApiError, getDevUser, Me, setDevUser } from "@/lib/api";

type SessionState = {
  me: Me | null;
  loading: boolean;
  error: string | null;
  signIn: (email: string) => Promise<void>;
  signOut: () => void;
  refresh: () => Promise<void>;
};

const SessionContext = createContext<SessionState | null>(null);

export function useSession(): SessionState {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error("useSession outside SessionProvider");
  return ctx;
}

export function SessionProvider({ children }: { children: React.ReactNode }) {
  const [me, setMe] = useState<Me | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!getDevUser()) {
      setMe(null);
      setLoading(false);
      return;
    }
    try {
      setMe(await api.me());
      setError(null);
    } catch (e) {
      setMe(null);
      setError(e instanceof ApiError ? e.message : "Can't reach the Emo_Text server.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const signIn = useCallback(
    async (email: string) => {
      setDevUser(email.trim().toLowerCase());
      setLoading(true);
      await refresh();
    },
    [refresh],
  );

  const signOut = useCallback(() => {
    setDevUser(null);
    setMe(null);
  }, []);

  return (
    <SessionContext.Provider value={{ me, loading, error, signIn, signOut, refresh }}>
      {children}
    </SessionContext.Provider>
  );
}
