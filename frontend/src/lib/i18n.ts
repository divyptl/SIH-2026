/**
 * UI localisation.
 *
 * English is bundled; every other locale is a separate chunk fetched the first
 * time it is selected. `src/locales/en.json` is the source of truth; fill the
 * other files with `uv run python -m scripts.translate_locales` in `backend/`.
 */
import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'

import en from '#/locales/en.json'
import {
  DEFAULT_LANGUAGE,
  getLanguage,
  isSupportedLanguage,
} from '#/lib/languages'

const LANGUAGE_STORAGE_KEY = 'satquery-language'

// English is bundled above, so it is excluded from the lazily loaded chunks.
const localeLoaders = import.meta.glob<{ default: Record<string, unknown> }>([
  '../locales/*.json',
  '!../locales/en.json',
])

async function loadLocale(code: string) {
  if (i18n.hasResourceBundle(code, 'translation')) return
  const path = `../locales/${code}.json`
  // Languages without a locale file yet fall back to English strings.
  if (!(path in localeLoaders)) return
  const { default: resources } = await localeLoaders[path]()
  i18n.addResourceBundle(code, 'translation', resources, true, true)
}

function readStoredLanguage(): string | null {
  try {
    return localStorage.getItem(LANGUAGE_STORAGE_KEY)
  } catch {
    // Storage can throw in private mode or when cookies are blocked.
    return null
  }
}

/** Stored choice first, then the browser's preferred languages, then English. */
function initialLanguage(): string {
  const stored = readStoredLanguage()
  if (stored && isSupportedLanguage(stored)) return stored
  for (const tag of navigator.languages) {
    const base = tag.toLowerCase().split('-')[0]
    if (isSupportedLanguage(base)) return base
  }
  return DEFAULT_LANGUAGE
}

function applyToDocument(code: string) {
  document.documentElement.lang = code
  document.documentElement.dir = getLanguage(code).dir
}

export async function changeLanguage(code: string) {
  if (!isSupportedLanguage(code)) return
  await loadLocale(code)
  await i18n.changeLanguage(code)
  try {
    localStorage.setItem(LANGUAGE_STORAGE_KEY, code)
  } catch {
    // Not persisting is fine; the choice still applies for this session.
  }
}

i18n.on('languageChanged', applyToDocument)

void i18n.use(initReactI18next).init({
  resources: { en: { translation: en } },
  lng: DEFAULT_LANGUAGE,
  fallbackLng: DEFAULT_LANGUAGE,
  interpolation: { escapeValue: false },
  returnNull: false,
})
applyToDocument(DEFAULT_LANGUAGE)

/** Resolves once the initial locale is loaded, so the first paint is localised. */
export const i18nReady: Promise<void> = changeLanguage(initialLanguage())

export default i18n
