"use client";

import { useAuth, useClerk } from "@clerk/nextjs";
import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { api, ApiError, getDevUser, Me, setDevUser, setTokenGetter } from "@/lib/api";
import { CLERK_ENABLED } from "@/lib/auth-config";

type SessionState = {
  mode: "clerk" | "dev";
  me: Me | null;
  loading: boolean;
  error: string | null;
  /** Dev mode only: Clerk signs in through its own UI. */
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
  return CLERK_ENABLED ? (
    <ClerkSession>{children}</ClerkSession>
  ) : (
    <DevSession>{children}</DevSession>
  );
}

/** Loads /v1/me whenever `signedIn` turns true. */
function useMe(signedIn: boolean | undefined) {
  const [me, setMe] = useState<Me | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!signedIn) {
      setMe(null);
      setLoading(signedIn === undefined);
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
  }, [signedIn]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { me, setMe, loading, setLoading, error, refresh };
}

function ClerkSession({ children }: { children: React.ReactNode }) {
  const { isLoaded, isSignedIn, getToken } = useAuth();
  const clerk = useClerk();
  const [ready, setReady] = useState(false);

  useEffect(() => {
    setTokenGetter(() => getToken());
    setReady(true);
    return () => setTokenGetter(null);
  }, [getToken]);

  const { me, loading, error, refresh } = useMe(isLoaded && ready ? isSignedIn : undefined);
  const signOut = useCallback(() => {
    clerk.signOut({ redirectUrl: "/" });
  }, [clerk]);
  const signIn = useCallback(async () => clerk.openSignIn(), [clerk]);

  return (
    <SessionContext.Provider value={{ mode: "clerk", me, loading, error, signIn, signOut, refresh }}>
      {children}
    </SessionContext.Provider>
  );
}

function DevSession({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<string | null | undefined>(undefined);
  useEffect(() => setUser(getDevUser()), []);
  const { me, loading, error, refresh } = useMe(user === undefined ? undefined : Boolean(user));

  const signIn = useCallback(async (email: string) => {
    const clean = email.trim().toLowerCase();
    setDevUser(clean);
    setUser(clean);
  }, []);

  const signOut = useCallback(() => {
    setDevUser(null);
    setUser(null);
  }, []);

  return (
    <SessionContext.Provider value={{ mode: "dev", me, loading, error, signIn, signOut, refresh }}>
      {children}
    </SessionContext.Provider>
  );
}
