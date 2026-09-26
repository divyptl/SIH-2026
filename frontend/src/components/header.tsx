import { LanguagePicker } from '#/components/language-picker'
import { ModeToggle } from '#/components/mode-toggle'

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
  return (
    <header className="sticky top-0 z-30 border-b bg-background/80 backdrop-blur">
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
          <LanguagePicker />
          <ModeToggle />
        </div>
      </div>
    </header>
  )
}

export default Header
