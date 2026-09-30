import type { Metadata, Viewport } from 'next';
import { DM_Sans, Space_Grotesk } from 'next/font/google';
import './globals.css';

const bodyFont = DM_Sans({ subsets: ['latin'], variable: '--font-body', display: 'swap' });
const displayFont = Space_Grotesk({ subsets: ['latin'], variable: '--font-display', display: 'swap' });

export const metadata: Metadata = {
  title: 'Delhi AirCast — Know the air before you go',
  description: 'Station-level short-term air quality forecasting across Delhi, with transparent historical context.',
  applicationName: 'Delhi AirCast',
};

export const viewport: Viewport = { themeColor: '#1b2340', width: 'device-width', initialScale: 1 };

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={`${bodyFont.variable} ${displayFont.variable}`}>{children}</body>
    </html>
  );
}
