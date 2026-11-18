import type { NextConfig } from "next";

// The browser talks to /api/*, which Next forwards to the FastAPI backend. Same origin, so no
// CORS, and signed audio links work as plain <audio src>.
const backend = process.env.EMOTEXT_API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  // Pin the workspace root so a stray lockfile in a parent folder is not picked up.
  turbopack: { root: __dirname },
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${backend}/:path*` }];
  },
};

export default nextConfig;
