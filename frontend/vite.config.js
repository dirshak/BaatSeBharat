import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The frontend fetches only static JSON under public/data/ (see
// src/dataClient.js) -- no dev-server proxy to a live backend needed.
// `npm run dev` reads directly from frontend/public/data/, regenerated
// locally via `python scripts/export_static_data.py`.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
  },
})
