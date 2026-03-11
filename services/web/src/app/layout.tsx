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
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Recursive:slnt,wght,CASL,CRSV,MONO@-15..0,300..1000,0..1,0..1,0..1&display=swap"
          rel="stylesheet"
        />
        <style>{`
          @font-face {
            font-family: 'Bitcount Prop Double Ink';
            src: url('/fonts/Bitcount_Prop_Double_Ink,Bitcount_Prop_Single_Ink/Bitcount_Prop_Double_Ink/BitcountPropDoubleInk-VariableFont_CRSV,ELSH,ELXP,SZP1,SZP2,XPN1,XPN2,YPN1,YPN2,slnt,wght.ttf') format('truetype');
            font-weight: 100 900;
            font-display: swap;
          }
          @font-face {
            font-family: 'Bitcount Prop Single Ink';
            src: url('/fonts/Bitcount_Prop_Double_Ink,Bitcount_Prop_Single_Ink/Bitcount_Prop_Single_Ink/BitcountPropSingleInk-VariableFont_CRSV,ELSH,ELXP,SZP1,SZP2,XPN1,XPN2,YPN1,YPN2,slnt,wght.ttf') format('truetype');
            font-weight: 100 900;
            font-display: swap;
          }
        `}</style>
      </head>
      <body style={{ margin: 0, fontFamily: "system-ui, sans-serif", backgroundColor: "#2b2b2b", color: "#e0e0e0" }}>
        {children}
      </body>
    </html>
  );
}
