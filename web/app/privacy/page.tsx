import type { Metadata } from "next";

export const metadata: Metadata = { title: "Privacy — Emo_Text" };

// Plain-language draft for the beta. Review before launch, and list your actual processors.
export default function Privacy() {
  return (
    <article className="prose">
      <h1>Privacy</h1>
      <p className="muted small">Beta draft · last updated November 2026</p>

      <h2>What we keep</h2>
      <ul>
        <li>Your account: the email and sign-in details handled by our login provider.</li>
        <li>Your uploads and the audio made from them, until you delete the book.</li>
        <li>Your credit history and payments. Card details stay with our payment provider.</li>
        <li>Error reports, which never include the text of your books.</li>
      </ul>

      <h2>Who processes it</h2>
      <p>
        To make the audio, the text of your book is sent to the speech engine you choose and,
        if you pick the AI Director, to an AI model that decides who is speaking. These providers
        process it to return the result. We don&apos;t sell your data or use your books to train models.
      </p>

      <h2>Deleting</h2>
      <p>
        Deleting a book removes the upload and its audio. To close your account and erase
        everything, contact us.
      </p>
    </article>
  );
}
