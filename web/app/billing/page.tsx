"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { useSession } from "@/components/Session";
import { api, ApiError, formatWords, Options } from "@/lib/api";

type Entry = Awaited<ReturnType<typeof api.ledger>>[number];

const REASONS: Record<string, string> = {
  signup: "Welcome credits",
  render: "Audiobook render",
  refund: "Refund (render failed)",
  purchase: "Purchase",
};

export default function BillingPage() {
  return (
    <Suspense>
      <Billing />
    </Suspense>
  );
}

function Billing() {
  const status = useSearchParams().get("status");
  const { me, refresh } = useSession();
  const [options, setOptions] = useState<Options | null>(null);
  const [ledger, setLedger] = useState<Entry[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!me) return;
    api.options().then(setOptions);
    api.ledger().then(setLedger);
  }, [me]);

  useEffect(() => {
    if (status === "success") refresh(); // the webhook may already have granted the credits
  }, [status, refresh]);

  async function buy(pack: string) {
    setError(null);
    try {
      window.location.href = (await api.checkout(pack)).url;
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  }

  if (!me) return <p className="muted">Sign in to manage credits.</p>;
  return (
    <>
      <h1>Credits</h1>
      <p className="lede">
        You have <strong>{formatWords(me.credits)}</strong> credits. One credit renders one word;
        previews are free, and failed renders are refunded.
      </p>
      {status === "success" && <p className="ok">Payment received, thank you.</p>}
      {status === "cancelled" && <p className="muted">Checkout cancelled. Nothing was charged.</p>}
      {error && <p className="error">{error}</p>}

      <div className="packs">
        {options?.packs.map((p) => (
          <div key={p.id} className="card pack">
            <strong>{formatWords(p.credits)} credits</strong>
            <span className="muted small">
              about {Math.round(p.credits / 90_000)} novel{p.credits >= 135_000 ? "s" : ""}
            </span>
            <button className="primary" disabled={!options.payments} onClick={() => buy(p.id)}>
              ${(p.price_cents / 100).toFixed(2)}
            </button>
          </div>
        ))}
      </div>
      {options && !options.payments && (
        <p className="small muted">Payments are not set up on this server yet.</p>
      )}

      <h2>History</h2>
      <table className="ledger">
        <tbody>
          {ledger.map((e, i) => (
            <tr key={i}>
              <td className="muted small">{new Date(e.created_at).toLocaleString()}</td>
              <td>{REASONS[e.reason] ?? e.reason}</td>
              <td className={e.delta < 0 ? "neg" : "pos"}>
                {e.delta > 0 ? "+" : ""}
                {formatWords(e.delta)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
