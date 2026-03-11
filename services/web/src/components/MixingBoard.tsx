"use client";

import { PrecisionSettings } from "@/lib/api";

interface MixingBoardProps {
  precision: PrecisionSettings;
  onChange: (precision: PrecisionSettings) => void;
}

/**
 * Precision-weighted mixing board.
 * Sliders for adjusting Logic and EQ module influence.
 */
export default function MixingBoard({ precision, onChange }: MixingBoardProps) {
  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Mixing Board</h3>
      <p style={styles.subtitle}>
        Adjust module precision weights to control reasoning emphasis
      </p>

      <div style={styles.sliderGroup}>
        <label style={styles.label}>
          <span style={styles.labelText}>Logic</span>
          <span style={styles.value}>{precision.logic.toFixed(1)}</span>
        </label>
        <input
          type="range"
          min="0.1"
          max="10"
          step="0.1"
          value={precision.logic}
          onChange={(e) =>
            onChange({ ...precision, logic: parseFloat(e.target.value) })
          }
          style={styles.slider}
        />
      </div>

      <div style={styles.sliderGroup}>
        <label style={styles.label}>
          <span style={styles.labelText}>EQ</span>
          <span style={styles.value}>{precision.eq.toFixed(1)}</span>
        </label>
        <input
          type="range"
          min="0.1"
          max="10"
          step="0.1"
          value={precision.eq}
          onChange={(e) =>
            onChange({ ...precision, eq: parseFloat(e.target.value) })
          }
          style={styles.slider}
        />
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: "16px",
    borderBottom: "1px solid #3a3a3a",
  },
  title: {
    margin: "0 0 4px 0",
    fontSize: "14px",
    fontWeight: 600,
    color: "#e0e0e0",
  },
  subtitle: {
    margin: "0 0 12px 0",
    fontSize: "12px",
    color: "#777",
  },
  sliderGroup: {
    marginBottom: "12px",
  },
  label: {
    display: "flex",
    justifyContent: "space-between",
    marginBottom: "4px",
    fontSize: "13px",
  },
  labelText: {
    fontWeight: 500,
  },
  value: {
    fontFamily: "monospace",
    color: "#e87b5f",
  },
  slider: {
    width: "100%",
    cursor: "pointer",
  },
};
