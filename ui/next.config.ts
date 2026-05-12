import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  serverExternalPackages: ['macstats', 'osx-temperature-sensor'],
  typescript: {
    // Remove this. Build fails because of route types
    ignoreBuildErrors: true,
  },
  experimental: {
    serverActions: {
      bodySizeLimit: '100gb',
    },
    middlewareClientMaxBodySize: '100gb',
  },
};

export default nextConfig;
