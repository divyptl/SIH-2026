import { Outlet, createRootRoute, useRouterState } from '@tanstack/react-router'

import { TanStackRouterDevtoolsPanel } from '@tanstack/react-router-devtools'
import { TanStackDevtools } from '@tanstack/react-devtools'

import { MotionConfig } from 'motion/react'

import '../styles.css'
import Header from '#/components/header'
import { ThemeProvider } from '#/components/theme-provider'

export const Route = createRootRoute({
  component: RootComponent,
})

function RootComponent() {
  // The devtools button would sit on the demo's presenter controls.
  const onDemo = useRouterState({
    select: (state) => state.location.pathname === '/demo',
  })
  return (
    <ThemeProvider defaultTheme="system">
      <MotionConfig reducedMotion="user">
        <Header />
        <Outlet />
      </MotionConfig>
      {!onDemo && (
        <TanStackDevtools
          config={{
            position: 'bottom-right',
          }}
          plugins={[
            {
              name: 'TanStack Router',
              render: <TanStackRouterDevtoolsPanel />,
            },
          ]}
        />
      )}
    </ThemeProvider>
  )
}
