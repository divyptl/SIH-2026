import * as React from 'react'

type Theme = 'dark' | 'light' | 'system'
type ResolvedTheme = 'dark' | 'light'

/**
 * Keep `THEME_STORAGE_KEY` in sync with the inline anti-flash script in
 * `index.html` — that script reads the same key before first paint.
 */
const THEME_STORAGE_KEY = 'vite-ui-theme'
const COLOR_SCHEME_QUERY = '(prefers-color-scheme: dark)'

type ThemeProviderState = {
  /** What the user picked, including `system`. */
  theme: Theme
  /** What is actually on the document — `system` resolved against the OS. */
  resolvedTheme: ResolvedTheme
  setTheme: (theme: Theme) => void
}

const ThemeProviderContext = React.createContext<
  ThemeProviderState | undefined
>(undefined)

function getSystemTheme(): ResolvedTheme {
  return window.matchMedia(COLOR_SCHEME_QUERY).matches ? 'dark' : 'light'
}

function readStoredTheme(storageKey: string, fallback: Theme): Theme {
  try {
    const stored = localStorage.getItem(storageKey)
    return stored === 'light' || stored === 'dark' || stored === 'system'
      ? stored
      : fallback
  } catch {
    // Storage can throw in private mode or when cookies are blocked.
    return fallback
  }
}

function ThemeProvider({
  children,
  defaultTheme = 'system',
  storageKey = THEME_STORAGE_KEY,
}: {
  children: React.ReactNode
  defaultTheme?: Theme
  storageKey?: string
}) {
  const [theme, setThemeState] = React.useState<Theme>(() =>
    readStoredTheme(storageKey, defaultTheme),
  )
  const [systemTheme, setSystemTheme] =
    React.useState<ResolvedTheme>(getSystemTheme)

  const resolvedTheme = theme === 'system' ? systemTheme : theme

  // Follow the OS preference live, so `system` flips without a reload.
  React.useEffect(() => {
    const media = window.matchMedia(COLOR_SCHEME_QUERY)
    const onChange = () => setSystemTheme(media.matches ? 'dark' : 'light')

    media.addEventListener('change', onChange)
    return () => media.removeEventListener('change', onChange)
  }, [])

  React.useEffect(() => {
    const root = document.documentElement

    root.classList.remove('light', 'dark')
    root.classList.add(resolvedTheme)
  }, [resolvedTheme])

  // Mirror the choice into other open tabs.
  React.useEffect(() => {
    const onStorage = (event: StorageEvent) => {
      if (event.key !== storageKey) return
      setThemeState(readStoredTheme(storageKey, defaultTheme))
    }

    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [storageKey, defaultTheme])

  const setTheme = React.useCallback(
    (next: Theme) => {
      try {
        localStorage.setItem(storageKey, next)
      } catch {
        // Not persisting is recoverable; the in-memory choice still applies.
      }
      setThemeState(next)
    },
    [storageKey],
  )

  const value = React.useMemo(
    () => ({ theme, resolvedTheme, setTheme }),
    [theme, resolvedTheme, setTheme],
  )

  return (
    <ThemeProviderContext.Provider value={value}>
      {children}
    </ThemeProviderContext.Provider>
  )
}

function useTheme() {
  const context = React.useContext(ThemeProviderContext)

  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }

  return context
}

export { ThemeProvider, useTheme, THEME_STORAGE_KEY }
export type { Theme, ResolvedTheme }
