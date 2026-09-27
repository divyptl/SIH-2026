import { defineConfig, loadEnv } from 'vite'
import { devtools } from '@tanstack/devtools-vite'

import { tanstackRouter } from '@tanstack/router-plugin/vite'

import viteReact from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const config = defineConfig(({ mode }) => {
  // index.html builds its link-preview URLs from VITE_SITE_URL. Without it
  // they fall back to site-relative paths rather than a literal placeholder.
  if (!loadEnv(mode, process.cwd(), 'VITE_').VITE_SITE_URL) {
    process.env.VITE_SITE_URL = ''
  }

  return {
    resolve: { tsconfigPaths: true },
    plugins: [
      devtools(),
      tailwindcss(),
      tanstackRouter({ target: 'react', autoCodeSplitting: true }),
      viteReact(),
    ],
  }
})

export default config
