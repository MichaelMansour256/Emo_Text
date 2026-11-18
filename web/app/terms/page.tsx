import type { Metadata } from "next";

export const metadata: Metadata = { title: "Terms — Emo_Text" };

// Plain-language draft for the beta. Have a lawyer review it before charging real customers.
export default function Terms() {
  return (
    <article className="prose">
      <h1>Terms of use</h1>
      <p className="muted small">Beta draft · last updated November 2026</p>

      <h2>Your books</h2>
      <p>
        Only upload books you wrote, books in the public domain, or books whose rights holder has
        allowed you to make an audio version. You keep every right you have in your book and in
        the audio made from it. We use your files only to make your audiobook.
      </p>

      <h2>What you may not do</h2>
      <ul>
        <li>Upload books you don&apos;t have the rights to, or distribute audio you aren&apos;t allowed to.</li>
        <li>Use the service to imitate a real person&apos;s voice, or for harassment, fraud or illegal content.</li>
        <li>Try to get around credit limits, upload limits or other safeguards.</li>
      </ul>
      <p>We may remove content and close accounts that break these rules.</p>

      <h2>AI voices</h2>
      <p>
        Narration is synthetic. If you publish it, say so where the store requires it (for example
        &ldquo;narrated by AI voices&rdquo;). Speaker and emotion choices are made automatically and can be wrong.
      </p>

      <h2>Credits and payments</h2>
      <p>
        One credit renders one word. Previews are free, and credits for a failed render come back
        automatically. Purchased credits don&apos;t expire during the beta and are not refundable for
        cash, except where the law requires it.
      </p>

      <h2>The beta</h2>
      <p>
        The service is in beta and provided as is. Keep your own copy of your books and of any
        audio you want to keep: we may delete files of inactive projects after 90 days.
      </p>

      <h2>Contact</h2>
      <p>Questions or a rights complaint: reply to any email from us, or contact the site owner.</p>
    </article>
  );
}
