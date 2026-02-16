// web/app/page.tsx
import React from "react";

async function getJson(path: string) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) return null;
  return res.json();
}

export default async function Home() {
  const signal = await getJson("http://localhost:3000/data/trade_signal.json");
  const state  = await getJson("http://localhost:3000/data/state.json");
  const snap   = await getJson("http://localhost:3000/data/universe_snapshot.json");

  const action = signal?.action ?? "N/A";
  const badge =
    action === "BUY" ? "bg-green-600" :
    action === "SELL" ? "bg-red-600" :
    "bg-gray-600";

  return (
    <main style={{ fontFamily: "system-ui", padding: 24, maxWidth: 1100, margin: "0 auto" }}>
      <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 6 }}>Merry Signal LIVE</h1>
      <div style={{ opacity: 0.75, marginBottom: 18 }}>
        asof: <b>{signal?.asof ?? state?.asof ?? "?"}</b>
      </div>

      <div style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: 16,
        marginBottom: 18
      }}>
        <div style={{ border: "1px solid #ddd", borderRadius: 12, padding: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
            <span className={badge} style={{
              color: "white", padding: "6px 10px", borderRadius: 999, fontWeight: 700
            }}>
              {action}
            </span>
            <div style={{ fontSize: 18, fontWeight: 800 }}>
              {signal?.target ?? "NO POSITION"}
            </div>
          </div>

          <div style={{ lineHeight: 1.9 }}>
            <div>LEV: <b>{signal?.lev ?? state?.lev ?? 3.0}</b></div>
            <div>Entry: <b>{signal?.entry_price ?? state?.entry_price ?? "-"}</b></div>
            <div>Stop (PrevWeekLow): <b>{signal?.stop_price ?? state?.stop_price ?? "-"}</b></div>
            <div>Equity (Realized): <b>{signal?.equity_realized ?? state?.realized_equity ?? 1.0}</b></div>
            <div>Equity (MTM): <b>{signal?.equity_mtm ?? "-"}</b></div>
          </div>
        </div>

        <div style={{ border: "1px solid #ddd", borderRadius: 12, padding: 16 }}>
          <div style={{ fontSize: 16, fontWeight: 800, marginBottom: 8 }}>Reason</div>
          <div style={{ whiteSpace: "pre-wrap", opacity: 0.9 }}>
            {signal?.reason ?? "—"}
          </div>
          <hr style={{ margin: "14px 0", border: "none", borderTop: "1px solid #eee" }} />
          <div style={{ fontSize: 16, fontWeight: 800, marginBottom: 6 }}>State</div>
          <pre style={{ background: "#fafafa", padding: 12, borderRadius: 10, overflowX: "auto" }}>
            {JSON.stringify(state, null, 2)}
          </pre>
        </div>
      </div>

      <div style={{ border: "1px solid #ddd", borderRadius: 12, padding: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
          <div style={{ fontSize: 16, fontWeight: 800 }}>Universe Snapshot</div>
          <div style={{ opacity: 0.7 }}>({snap?.rows?.length ?? 0} tickers)</div>
        </div>

        <div style={{ overflowX: "auto", marginTop: 10 }}>
          <table style={{ borderCollapse: "collapse", width: "100%" }}>
            <thead>
              <tr style={{ textAlign: "left", borderBottom: "1px solid #eee" }}>
                {["ticker","close","zone","WaveOK","FirstBreak","Retest","Rebreak","Exit","ATR%"].map(h => (
                  <th key={h} style={{ padding: "8px 10px", fontSize: 12, opacity: 0.7 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(snap?.rows ?? []).map((r: any) => (
                <tr key={r.ticker} style={{ borderBottom: "1px solid #f2f2f2" }}>
                  <td style={{ padding: "8px 10px", fontWeight: 700 }}>{r.ticker}</td>
                  <td style={{ padding: "8px 10px" }}>{r.close?.toFixed?.(2) ?? r.close}</td>
                  <td style={{ padding: "8px 10px" }}>{r.zone?.toFixed?.(2) ?? "-"}</td>
                  <td style={{ padding: "8px 10px" }}>{String(r.wave_ok)}</td>
                  <td style={{ padding: "8px 10px" }}>{String(r.first_break)}</td>
                  <td style={{ padding: "8px 10px" }}>{String(r.retest)}</td>
                  <td style={{ padding: "8px 10px" }}>{String(r.rebreak)}</td>
                  <td style={{ padding: "8px 10px" }}>{String(r.exit)}</td>
                  <td style={{ padding: "8px 10px" }}>
                    {r.atr_pct == null ? "-" : (r.atr_pct * 100).toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div style={{ marginTop: 10, opacity: 0.7, fontSize: 12 }}>
          data source: <code>/public/data/*.json</code>
        </div>
      </div>
    </main>
  );
}
