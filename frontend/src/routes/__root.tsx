import { Outlet, createRootRoute } from '@tanstack/react-router'

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
  return (
    <ThemeProvider defaultTheme="system">
      <MotionConfig reducedMotion="user">
        <Header />
        <Outlet />
      </MotionConfig>
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
    </ThemeProvider>
  )
}
