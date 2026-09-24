import * as React from 'react'
import { ChevronsUpDownIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

import { Button } from '#/components/ui/button'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from '#/components/ui/command'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '#/components/ui/popover'
import { Spinner } from '#/components/ui/spinner'
import { changeLanguage } from '#/lib/i18n'
import { LANGUAGES, getLanguage, isSupportedLanguage } from '#/lib/languages'
import type { Language } from '#/lib/languages'

const RECENT_STORAGE_KEY = 'satquery-recent-languages'
const MAX_SUGGESTED = 4

// Alphabetical by English name, so the full list reads the same in every UI
// language; English itself leads because it is the models' own language.
const SORTED = [...LANGUAGES].sort((a, b) =>
  a.code === 'en' ? -1 : b.code === 'en' ? 1 : a.name.localeCompare(b.name),
)

function readRecent(): Array<string> {
  try {
    const stored: unknown = JSON.parse(
      localStorage.getItem(RECENT_STORAGE_KEY) ?? '[]',
    )
    return Array.isArray(stored)
      ? stored.filter((code): code is string => typeof code === 'string')
      : []
  } catch {
    // Storage can be unavailable (private mode) or hold junk; start fresh.
    return []
  }
}

function rememberRecent(code: string) {
  try {
    const next = [code, ...readRecent().filter((c) => c !== code)].slice(0, 3)
    localStorage.setItem(RECENT_STORAGE_KEY, JSON.stringify(next))
  } catch {
    // Not remembering is harmless; suggestions just fall back to the browser.
  }
}

/** Current language, then recent picks, then the browser's preferences. */
function suggestedLanguages(current: string): Array<Language> {
  const browser = navigator.languages.map(
    (tag) => tag.toLowerCase().split('-')[0],
  )
  const codes = [current, ...readRecent(), ...browser, 'hi', 'en']
  return [...new Set(codes)]
    .filter(isSupportedLanguage)
    .slice(0, MAX_SUGGESTED)
    .map((code) => getLanguage(code))
}

function LanguageRow({
  language,
  isCurrent,
}: {
  language: Language
  isCurrent: boolean
}) {
  return (
    <>
      <span
        aria-hidden
        lang={language.code}
        className="flex size-7 shrink-0 items-center justify-center rounded-md border bg-background text-sm leading-none font-medium group-data-[checked=true]/command-item:border-foreground group-data-[checked=true]/command-item:bg-foreground group-data-[checked=true]/command-item:text-background"
      >
        {language.glyph}
      </span>
      {/* flex-1 takes the free space, so the check always sits at the end. */}
      <span className="flex min-w-0 flex-1 items-baseline justify-between gap-3">
        <span
          lang={language.code}
          dir={language.dir}
          className={isCurrent ? 'font-medium' : undefined}
        >
          {language.nativeName}
        </span>
        {language.nativeName !== language.name && (
          <span lang="en" className="text-xs text-muted-foreground">
            {language.name}
          </span>
        )}
      </span>
    </>
  )
}

function LanguagePicker() {
  const { t, i18n } = useTranslation()
  const current = getLanguage(i18n.resolvedLanguage)
  const [open, setOpen] = React.useState(false)
  const [search, setSearch] = React.useState('')
  const [pending, setPending] = React.useState<string | null>(null)

  // Recomputed on open so a pick made moments ago shows up straight away.
  const suggested = React.useMemo(
    () => (open ? suggestedLanguages(current.code) : []),
    [open, current.code],
  )

  const select = async (code: string) => {
    setOpen(false)
    if (code === current.code) return
    rememberRecent(code)
    // Other locales are lazy-loaded chunks; show progress while one arrives.
    setPending(code)
    try {
      await changeLanguage(code)
    } finally {
      setPending(null)
    }
  }

  const item = (language: Language, keyPrefix = '') => (
    <CommandItem
      key={keyPrefix + language.code}
      // Unique per group, but searchable by every name the user might type.
      value={keyPrefix + language.code}
      keywords={[language.name, language.nativeName, language.code]}
      data-checked={language.code === current.code}
      onSelect={() => void select(language.code)}
      className="gap-2.5 py-1.5"
    >
      <LanguageRow
        language={language}
        isCurrent={language.code === current.code}
      />
    </CommandItem>
  )

  return (
    <Popover
      open={open}
      onOpenChange={(next) => {
        setOpen(next)
        if (!next) setSearch('')
      }}
    >
      <PopoverTrigger
        render={
          <Button
            variant="outline"
            aria-label={`${t('header.language')}: ${current.nativeName}`}
            className="gap-2 ps-1.5"
          />
        }
      >
        <span
          aria-hidden
          lang={current.code}
          className="flex size-5 items-center justify-center rounded bg-foreground text-[11px] leading-none font-medium text-background"
        >
          {pending ? <Spinner className="size-3" /> : current.glyph}
        </span>
        <span lang={current.code} className="max-sm:hidden">
          {current.nativeName}
        </span>
        <ChevronsUpDownIcon className="size-3.5 text-muted-foreground" />
      </PopoverTrigger>
      <PopoverContent align="end" className="w-72 p-0">
        <Command>
          <CommandInput
            value={search}
            onValueChange={setSearch}
            placeholder={t('header.searchLanguage')}
            aria-label={t('header.searchLanguage')}
          />
          <CommandList className="max-h-[min(24rem,60vh)]">
            <CommandEmpty>{t('header.noLanguage')}</CommandEmpty>
            {/* While searching, one flat list of matches is clearer. */}
            {search === '' && (
              <>
                <CommandGroup heading={t('header.suggested')}>
                  {suggested.map((language) => item(language, 'suggested-'))}
                </CommandGroup>
                <CommandSeparator />
              </>
            )}
            <CommandGroup
              heading={search === '' ? t('header.allLanguages') : undefined}
            >
              {SORTED.map((language) => item(language))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}

export { LanguagePicker }
