import { useTranslation } from 'react-i18next'

/** Page heading: what the tool does, in the user's words. */
export function Hero() {
  const { t } = useTranslation()

  return (
    <section className="mx-auto max-w-6xl px-4 pt-10 pb-8 sm:px-6 sm:pt-14">
      <h1 className="max-w-3xl font-heading text-4xl leading-[1.05] font-semibold tracking-[-0.02em] text-balance sm:text-5xl">
        {t('home.title')}
      </h1>
      <p className="mt-4 max-w-2xl text-base/7 text-pretty text-muted-foreground">
        {t('home.subtitle')}
      </p>
    </section>
  )
}
