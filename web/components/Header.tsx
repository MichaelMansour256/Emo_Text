"use client";

import Link from "next/link";
import { formatWords } from "@/lib/api";
import { useSession } from "./Session";

export function Header() {
  const { me, signOut } = useSession();
  return (
    <header className="topbar">
      <Link href="/" className="brand">
        <span className="brand-mark" aria-hidden>
          ◖◗
        </span>
        Emo_Text
      </Link>
      {me && (
        <nav className="topnav">
          <Link href="/billing" className="credits" title="1 credit = 1 word rendered">
            {formatWords(me.credits)} credits
          </Link>
          <span className="muted small">{me.email}</span>
          <button className="link" onClick={signOut}>
            Sign out
          </button>
        </nav>
      )}
    </header>
  );
}
