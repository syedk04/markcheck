import { defineConfig } from 'vitest/config'
import preact from '@preact/preset-vite'

export default defineConfig({
  base: '/markcheck/',
  plugins: [preact()],
  worker: {
    format: 'es',
  },
  optimizeDeps: {
    exclude: ['@xenova/transformers'],
  },
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
  },
})
