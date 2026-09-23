import type { Metadata, Viewport } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Delhi AirCast — Know the air before you go',
  description: 'Station-level short-term air quality forecasting across Delhi, with transparent historical context.',
  applicationName: 'Delhi AirCast',
};

export const viewport: Viewport = { themeColor: '#20352d', width: 'device-width', initialScale: 1 };

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
