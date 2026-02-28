import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import glsl from 'vite-plugin-glsl'

export default defineConfig({
  base: '/llm-sandbox/',
  plugins: [react(), glsl()],
  build: {
    target: 'esnext',
  },
  worker: {
    format: 'es',
  },
})
