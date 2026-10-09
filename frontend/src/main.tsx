import ReactDOM from 'react-dom/client'
import { RouterProvider, createRouter } from '@tanstack/react-router'
import { routeTree } from './routeTree.gen'
import { i18nReady } from '#/lib/i18n'

const router = createRouter({
  routeTree,
  defaultPreload: 'intent',
  scrollRestoration: true,
})

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}

const rootElement = document.getElementById('app')!

if (!rootElement.innerHTML) {
  // A failed locale fetch still renders, falling back to English.
  void i18nReady.finally(() => {
    const root = ReactDOM.createRoot(rootElement)
    root.render(<RouterProvider router={router} />)
  })
}
