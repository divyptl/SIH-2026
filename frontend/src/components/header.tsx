import { Link, useMatchRoute } from '@tanstack/react-router'
import { useTranslation } from 'react-i18next'

import { LanguagePicker } from '#/components/language-picker'
import { ModeToggle } from '#/components/mode-toggle'
import { Button } from '#/components/ui/button'
import { Separator } from '#/components/ui/separator'
import { buttonVariants } from '@/components/ui/button'

function Github() {
  return (
    <a
      href="https://github.com/divyptl/SIH-2026"
      target="_blank"
      rel="noopener noreferrer"
      className={buttonVariants({ variant: 'outline', size: 'icon' })}
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="16"
        height="16"
        fill="currentColor"
        class="bi bi-github"
        viewBox="0 0 16 16"
      >
        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8" />
      </svg>
    </a>
  )
}

/** A satellite's orbit around a point on the ground. */
function Mark() {
  return (
    <svg
      viewBox="0 0 24 24"
      aria-hidden
      className="size-6 shrink-0"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
    >
      <circle cx="12" cy="12" r="3.2" fill="currentColor" stroke="none" />
      <ellipse
        cx="12"
        cy="12"
        rx="10"
        ry="4.6"
        transform="rotate(-28 12 12)"
        opacity="0.55"
      />
      <circle cx="20.1" cy="7.3" r="1.7" fill="currentColor" stroke="none" />
    </svg>
  )
}

const Header = () => {
  const { t } = useTranslation()
  const onDemo = Boolean(useMatchRoute()({ to: '/demo' }))
  return (
    <header className="sticky top-0 z-30 bg-background/80 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-4 px-4 sm:px-6">
        <a
          href="/"
          lang="en"
          className="flex items-center gap-2 rounded-md font-heading text-[17px] font-semibold tracking-tight focus-visible:outline-2 focus-visible:outline-ring"
        >
          <Mark />
          SatQuery AI
        </a>
        <div className="flex items-center gap-2">
          <Button
            variant={onDemo ? 'secondary' : 'ghost'}
            nativeButton={false}
            render={<Link to="/demo" />}
            className="hidden sm:inline-flex"
          >
            {t('header.demo')}
          </Button>
          <LanguagePicker />
          <ModeToggle />
          <Github />
        </div>
      </div>
      <Separator />
    </header>
  )
}

export default Header
