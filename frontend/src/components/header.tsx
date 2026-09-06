import { ModeToggle } from '#/components/mode-toggle'
import { Typography } from '#/components/typography'

const Header = () => {
  return (
    <header className="flex items-center justify-between gap-4 border-b px-4 py-3">
      <Typography variant="h4">SatQuery AI</Typography>
      <ModeToggle />
    </header>
  )
}

export default Header
