import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  // Repo name: served at https://<user>.github.io/psr-moon/
  // Keep '/' for local dev; Pages build uses '/psr-moon/'.
  base: process.env.GITHUB_PAGES === 'true' ? '/psr-moon/' : '/',
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
