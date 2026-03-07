import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "CLSA",
  description: "Collaborative Latent Superposition Architecture",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body style={{ margin: 0, fontFamily: "system-ui, sans-serif" }}>
        {children}
      </body>
    </html>
  );
}
