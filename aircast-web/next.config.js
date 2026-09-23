/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  async rewrites() {
    return [{
      source: '/backend/:path*',
      destination: `${process.env.AIRCAST_API_URL || 'http://127.0.0.1:8000'}/:path*`,
    }];
  },
};

module.exports = nextConfig;
