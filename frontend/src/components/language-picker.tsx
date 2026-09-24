import { LanguagesIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

import { Button } from '#/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from '#/components/ui/dropdown-menu'
import { changeLanguage } from '#/lib/i18n'
import { LANGUAGES, getLanguage } from '#/lib/languages'

function LanguagePicker() {
  const { t, i18n } = useTranslation()
  const current = getLanguage(i18n.resolvedLanguage)

  return (
    <DropdownMenu>
      <DropdownMenuTrigger render={<Button variant="outline" />}>
        <LanguagesIcon data-icon="inline-start" />
        <span>{current.nativeName}</span>
        <span className="sr-only">{t('header.language')}</span>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        className="max-h-96 w-56 overflow-y-auto"
      >
        <DropdownMenuGroup>
          <DropdownMenuLabel>{t('header.language')}</DropdownMenuLabel>
          <DropdownMenuRadioGroup
            value={current.code}
            onValueChange={(code: string) => void changeLanguage(code)}
          >
            {LANGUAGES.map((language) => (
              <DropdownMenuRadioItem
                key={language.code}
                value={language.code}
                lang={language.code}
              >
                <span>{language.nativeName}</span>
                {language.nativeName !== language.name && (
                  <span
                    lang="en"
                    className="ms-auto me-5 text-xs text-muted-foreground"
                  >
                    {language.name}
                  </span>
                )}
              </DropdownMenuRadioItem>
            ))}
          </DropdownMenuRadioGroup>
        </DropdownMenuGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export { LanguagePicker }
