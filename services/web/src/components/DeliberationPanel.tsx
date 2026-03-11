"use client";

import { DeliberationMetadata } from "@/lib/api";

interface DeliberationPanelProps {
  deliberation: DeliberationMetadata;
}

/**
 * Displays deliberation metadata for a message.
 * Shows steps taken, convergence status, entropy, and per-module precisions.
 */
export default function DeliberationPanel({
  deliberation,
}: DeliberationPanelProps) {
  return (
    <div style={styles.container}>
      <div style={styles.header}>Deliberation</div>

      <div style={styles.grid}>
        <div style={styles.stat}>
          <span style={styles.statLabel}>Steps</span>
          <span style={styles.statValue}>{deliberation.steps}</span>
        </div>

        <div style={styles.stat}>
          <span style={styles.statLabel}>Converged</span>
          <span style={styles.statValue}>
            {deliberation.converged ? "Yes" : "No"}
          </span>
        </div>

        <div style={styles.stat}>
          <span style={styles.statLabel}>Entropy</span>
          <span style={styles.statValue}>
            {deliberation.final_entropy.toFixed(3)}
          </span>
        </div>

        {Object.entries(deliberation.module_precisions).map(([mod, val]) => (
          <div key={mod} style={styles.stat}>
            <span style={styles.statLabel}>{mod} precision</span>
            <span style={styles.statValue}>{val.toFixed(2)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    marginTop: "8px",
    padding: "8px 12px",
    backgroundColor: "#2b2b2b",
    borderRadius: "6px",
    fontSize: "12px",
  },
  header: {
    fontWeight: 600,
    marginBottom: "6px",
    color: "#999",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "4px 16px",
  },
  stat: {
    display: "flex",
    justifyContent: "space-between",
  },
  statLabel: {
    color: "#888",
  },
  statValue: {
    fontFamily: "monospace",
    fontWeight: 500,
    color: "#e0e0e0",
  },
};
