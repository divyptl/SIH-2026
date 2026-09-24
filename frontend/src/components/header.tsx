import { LanguagePicker } from '#/components/language-picker'
import { ModeToggle } from '#/components/mode-toggle'

const Header = () => {
  return (
    <header className="sticky top-0 z-30 border-b bg-background/80 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-4 px-4 sm:px-6">
        <a
          href="/"
          lang="en"
          className="rounded-md text-base font-semibold tracking-tight focus-visible:outline-2 focus-visible:outline-ring"
        >
          SatQuery AI
        </a>
        <div className="flex items-center gap-2">
          <LanguagePicker />
          <ModeToggle />
        </div>
      </div>
    </header>
  )
}

export default Header
