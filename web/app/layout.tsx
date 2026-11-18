import { ClerkProvider } from "@clerk/nextjs";
import type { Metadata } from "next";
import Link from "next/link";
import { Header } from "@/components/Header";
import { SessionProvider } from "@/components/Session";
import { CLERK_ENABLED } from "@/lib/auth-config";
import "./globals.css";

export const metadata: Metadata = {
  title: "Emo_Text — expressive audiobooks",
  description: "Turn a book into a full-cast, emotion-aware audiobook.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const app = (
    <SessionProvider>
      <Header />
      <main className="page">{children}</main>
      <footer className="footer small muted">
        <Link href="/terms">Terms</Link>
        <Link href="/privacy">Privacy</Link>
      </footer>
    </SessionProvider>
  );
  return (
    <html lang="en">
      <body>{CLERK_ENABLED ? <ClerkProvider>{app}</ClerkProvider> : app}</body>
    </html>
  );
}
