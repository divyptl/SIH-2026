/**
 * Languages the UI is localised into and a query can be written in.
 *
 * Codes must match `LANGUAGES` in `backend/services/translation.py` and the
 * file names in `src/locales/`.
 */

export interface Language {
  code: string
  /** English name, shown as secondary text in the picker. */
  name: string
  /** Endonym, shown as the primary text in the picker. */
  nativeName: string
  dir: 'ltr' | 'rtl'
}

export const LANGUAGES: Array<Language> = [
  { code: 'en', name: 'English', nativeName: 'English', dir: 'ltr' },
  { code: 'as', name: 'Assamese', nativeName: 'অসমীয়া', dir: 'ltr' },
  { code: 'bn', name: 'Bengali', nativeName: 'বাংলা', dir: 'ltr' },
  { code: 'brx', name: 'Bodo', nativeName: 'बड़ो', dir: 'ltr' },
  { code: 'doi', name: 'Dogri', nativeName: 'डोगरी', dir: 'ltr' },
  { code: 'gu', name: 'Gujarati', nativeName: 'ગુજરાતી', dir: 'ltr' },
  { code: 'hi', name: 'Hindi', nativeName: 'हिन्दी', dir: 'ltr' },
  { code: 'kn', name: 'Kannada', nativeName: 'ಕನ್ನಡ', dir: 'ltr' },
  { code: 'ks', name: 'Kashmiri', nativeName: 'کٲشُر', dir: 'rtl' },
  { code: 'gom', name: 'Konkani', nativeName: 'कोंकणी', dir: 'ltr' },
  { code: 'mai', name: 'Maithili', nativeName: 'मैथिली', dir: 'ltr' },
  { code: 'ml', name: 'Malayalam', nativeName: 'മലയാളം', dir: 'ltr' },
  { code: 'mni', name: 'Manipuri', nativeName: 'ꯃꯤꯇꯩꯂꯣꯟ', dir: 'ltr' },
  { code: 'mr', name: 'Marathi', nativeName: 'मराठी', dir: 'ltr' },
  { code: 'ne', name: 'Nepali', nativeName: 'नेपाली', dir: 'ltr' },
  { code: 'or', name: 'Odia', nativeName: 'ଓଡ଼ିଆ', dir: 'ltr' },
  { code: 'pa', name: 'Punjabi', nativeName: 'ਪੰਜਾਬੀ', dir: 'ltr' },
  { code: 'sa', name: 'Sanskrit', nativeName: 'संस्कृतम्', dir: 'ltr' },
  { code: 'sat', name: 'Santali', nativeName: 'ᱥᱟᱱᱛᱟᱲᱤ', dir: 'ltr' },
  { code: 'sd', name: 'Sindhi', nativeName: 'سنڌي', dir: 'rtl' },
  { code: 'ta', name: 'Tamil', nativeName: 'தமிழ்', dir: 'ltr' },
  { code: 'te', name: 'Telugu', nativeName: 'తెలుగు', dir: 'ltr' },
  { code: 'ur', name: 'Urdu', nativeName: 'اردو', dir: 'rtl' },
]

export const DEFAULT_LANGUAGE = 'en'

const BY_CODE = new Map(LANGUAGES.map((language) => [language.code, language]))

export function getLanguage(code: string | undefined): Language {
  return BY_CODE.get(code ?? '') ?? BY_CODE.get(DEFAULT_LANGUAGE)!
}

export function isSupportedLanguage(code: string): boolean {
  return BY_CODE.has(code)
}
