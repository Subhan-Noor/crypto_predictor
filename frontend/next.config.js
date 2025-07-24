/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'https://cryptopredictor-production.up.railway.app'
  },
  /* config options here */
}

module.exports = nextConfig 