import { useMemo } from "react";
import { PrecisionSettings } from "./api";

export interface Theme {
  // 0 = pure logic, 1 = pure EQ
  ratio: number;
  // Fonts
  fontFamily: string;
  fontVariationSettings: string;
  // Colors
  accent: string;
  accentHover: string;
  mainBg: string;
  surfaceBg: string;
  sidebarBg: string;
  border: string;
  text: string;
  textMuted: string;
  // Shape
  borderRadius: string;
  messageBorderRadius: string;
  // Gradient for name text (3 stops: light, mid, dark)
  nameGradientLight: string;
  nameGradientMid: string;
  nameGradientDark: string;
  // Content
  placeholder: string;
  letterSpacing: string;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpHex(hexA: string, hexB: string, t: number): string {
  const a = parseInt(hexA.slice(1), 16);
  const b = parseInt(hexB.slice(1), 16);

  const rA = (a >> 16) & 0xff, gA = (a >> 8) & 0xff, bA = a & 0xff;
  const rB = (b >> 16) & 0xff, gB = (b >> 8) & 0xff, bB = b & 0xff;

  const r = Math.round(lerp(rA, rB, t));
  const g = Math.round(lerp(gA, gB, t));
  const bl = Math.round(lerp(bA, bB, t));

  return `#${((r << 16) | (g << 8) | bl).toString(16).padStart(6, "0")}`;
}

// Logic theme (cool, precise, structured)
const LOGIC = {
  accent: "#5b9bf0",
  accentHover: "#4a8ae0",
  mainBg: "#252830",
  surfaceBg: "#2e3140",
  sidebarBg: "#1c1e28",
  border: "#3a3d4a",
  text: "#dce0e8",
  textMuted: "#7a7f8e",
};

// EQ theme (warm, soft, calming)
const EQ = {
  accent: "#f0a878",
  accentHover: "#e89860",
  mainBg: "#2d2828",
  surfaceBg: "#3a3232",
  sidebarBg: "#241e1e",
  border: "#4a3a3a",
  text: "#e8e0dc",
  textMuted: "#8e7a72",
};

export function useTheme(precision: PrecisionSettings): Theme {
  return useMemo(() => {
    const total = precision.eq + precision.logic;
    // Avoid division by zero; default to balanced
    const ratio = total > 0 ? precision.eq / total : 0.5;

    return {
      ratio,
      fontFamily: "'Recursive', sans-serif",
      fontVariationSettings: `'CASL' ${ratio}, 'wght' ${Math.round(lerp(400, 400, ratio))}`,
      accent: lerpHex(LOGIC.accent, EQ.accent, ratio),
      accentHover: lerpHex(LOGIC.accentHover, EQ.accentHover, ratio),
      mainBg: lerpHex(LOGIC.mainBg, EQ.mainBg, ratio),
      surfaceBg: lerpHex(LOGIC.surfaceBg, EQ.surfaceBg, ratio),
      sidebarBg: lerpHex(LOGIC.sidebarBg, EQ.sidebarBg, ratio),
      border: lerpHex(LOGIC.border, EQ.border, ratio),
      text: lerpHex(LOGIC.text, EQ.text, ratio),
      textMuted: lerpHex(LOGIC.textMuted, EQ.textMuted, ratio),
      borderRadius: `${Math.round(lerp(4, 14, ratio))}px`,
      messageBorderRadius: `${Math.round(lerp(4, 16, ratio))}px`,
      nameGradientLight: lerpHex("#7ec8e3", "#f0a878", ratio),
      nameGradientMid: lerpHex("#5b9bf0", "#e87b5f", ratio),
      nameGradientDark: lerpHex("#3a6fd8", "#d4566a", ratio),
      placeholder: ratio > 0.6
        ? "What's on your mind?"
        : ratio < 0.4
          ? "Ask a question..."
          : "Type a message...",
      letterSpacing: `${lerp(-0.02, 0.01, ratio)}em`,
    };
  }, [precision.eq, precision.logic]);
}
