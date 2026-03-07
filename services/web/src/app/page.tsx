"use client";

import { useState } from "react";
import ChatWindow from "@/components/ChatWindow";
import MixingBoard from "@/components/MixingBoard";
import { PrecisionSettings } from "@/lib/api";

export default function Home() {
  const [precision, setPrecision] = useState<PrecisionSettings>({
    logic: 1.0,
    eq: 1.0,
  });

  return (
    <div style={styles.layout}>
      {/* Sidebar */}
      <aside style={styles.sidebar}>
        <div style={styles.logo}>
          <h1 style={styles.logoText}>CLSA</h1>
          <p style={styles.logoSubtext}>
            Collaborative Latent Superposition Architecture
          </p>
        </div>
        <MixingBoard precision={precision} onChange={setPrecision} />
      </aside>

      {/* Main chat area */}
      <main style={styles.main}>
        <ChatWindow precision={precision} />
      </main>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  layout: {
    display: "flex",
    height: "100vh",
  },
  sidebar: {
    width: "280px",
    borderRight: "1px solid #e0e0e0",
    display: "flex",
    flexDirection: "column",
    backgroundColor: "#fafafa",
  },
  logo: {
    padding: "20px 16px",
    borderBottom: "1px solid #e0e0e0",
  },
  logoText: {
    margin: 0,
    fontSize: "20px",
    fontWeight: 700,
    letterSpacing: "0.05em",
  },
  logoSubtext: {
    margin: "4px 0 0 0",
    fontSize: "11px",
    color: "#888",
  },
  main: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
  },
};
