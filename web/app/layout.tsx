import type { Metadata } from "next";
import { Header } from "@/components/Header";
import { SessionProvider } from "@/components/Session";
import "./globals.css";

export const metadata: Metadata = {
  title: "Emo_Text — expressive audiobooks",
  description: "Turn a book into a full-cast, emotion-aware audiobook.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <SessionProvider>
          <Header />
          <main className="page">{children}</main>
        </SessionProvider>
      </body>
    </html>
  );
}
