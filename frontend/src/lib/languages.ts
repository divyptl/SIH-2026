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
  /** A letter of the language's script, shown as a tile in the picker. */
  glyph: string
}

export const LANGUAGES: Array<Language> = [
  {
    code: 'en',
    name: 'English',
    nativeName: 'English',
    dir: 'ltr',
    glyph: 'A',
  },
  {
    code: 'as',
    name: 'Assamese',
    nativeName: 'অসমীয়া',
    dir: 'ltr',
    glyph: 'অ',
  },
  { code: 'bn', name: 'Bengali', nativeName: 'বাংলা', dir: 'ltr', glyph: 'অ' },
  { code: 'brx', name: 'Bodo', nativeName: 'बड़ो', dir: 'ltr', glyph: 'अ' },
  { code: 'doi', name: 'Dogri', nativeName: 'डोगरी', dir: 'ltr', glyph: 'अ' },
  {
    code: 'gu',
    name: 'Gujarati',
    nativeName: 'ગુજરાતી',
    dir: 'ltr',
    glyph: 'અ',
  },
  { code: 'hi', name: 'Hindi', nativeName: 'हिन्दी', dir: 'ltr', glyph: 'अ' },
  { code: 'kn', name: 'Kannada', nativeName: 'ಕನ್ನಡ', dir: 'ltr', glyph: 'ಅ' },
  { code: 'ks', name: 'Kashmiri', nativeName: 'کٲشُر', dir: 'rtl', glyph: 'ک' },
  {
    code: 'gom',
    name: 'Konkani',
    nativeName: 'कोंकणी',
    dir: 'ltr',
    glyph: 'अ',
  },
  {
    code: 'mai',
    name: 'Maithili',
    nativeName: 'मैथिली',
    dir: 'ltr',
    glyph: 'अ',
  },
  {
    code: 'ml',
    name: 'Malayalam',
    nativeName: 'മലയാളം',
    dir: 'ltr',
    glyph: 'അ',
  },
  {
    code: 'mni',
    name: 'Manipuri',
    nativeName: 'ꯃꯤꯇꯩꯂꯣꯟ',
    dir: 'ltr',
    glyph: 'ꯑ',
  },
  { code: 'mr', name: 'Marathi', nativeName: 'मराठी', dir: 'ltr', glyph: 'अ' },
  { code: 'ne', name: 'Nepali', nativeName: 'नेपाली', dir: 'ltr', glyph: 'अ' },
  { code: 'or', name: 'Odia', nativeName: 'ଓଡ଼ିଆ', dir: 'ltr', glyph: 'ଅ' },
  { code: 'pa', name: 'Punjabi', nativeName: 'ਪੰਜਾਬੀ', dir: 'ltr', glyph: 'ਅ' },
  {
    code: 'sa',
    name: 'Sanskrit',
    nativeName: 'संस्कृतम्',
    dir: 'ltr',
    glyph: 'अ',
  },
  {
    code: 'sat',
    name: 'Santali',
    nativeName: 'ᱥᱟᱱᱛᱟᱲᱤ',
    dir: 'ltr',
    glyph: 'ᱚ',
  },
  { code: 'sd', name: 'Sindhi', nativeName: 'سنڌي', dir: 'rtl', glyph: 'س' },
  { code: 'ta', name: 'Tamil', nativeName: 'தமிழ்', dir: 'ltr', glyph: 'அ' },
  { code: 'te', name: 'Telugu', nativeName: 'తెలుగు', dir: 'ltr', glyph: 'అ' },
  { code: 'ur', name: 'Urdu', nativeName: 'اردو', dir: 'rtl', glyph: 'ع' },
]

export const DEFAULT_LANGUAGE = 'en'

const BY_CODE = new Map(LANGUAGES.map((language) => [language.code, language]))

export function getLanguage(code: string | undefined): Language {
  return BY_CODE.get(code ?? '') ?? BY_CODE.get(DEFAULT_LANGUAGE)!
}

export function isSupportedLanguage(code: string): boolean {
  return BY_CODE.has(code)
}
