import { useReducedMotion } from 'motion/react'
import { useTranslation } from 'react-i18next'

import BlurText from '#/components/BlurText'

/** Page heading; the title resolves out of blur once on load. */
export function Hero() {
  const { t, i18n } = useTranslation()
  const reduceMotion = useReducedMotion()
  const titleClass =
    'text-3xl leading-tight font-semibold tracking-tight text-balance sm:text-4xl'

  return (
    <section className="mx-auto max-w-6xl px-4 pt-12 pb-8 sm:px-6 sm:pt-16">
      {reduceMotion ? (
        <h1 className={titleClass}>{t('home.title')}</h1>
      ) : (
        <>
          <h1 className="sr-only">{t('home.title')}</h1>
          <div aria-hidden>
            <BlurText
              key={i18n.resolvedLanguage}
              text={t('home.title')}
              animateBy="words"
              direction="bottom"
              delay={80}
              className={titleClass}
            />
          </div>
        </>
      )}
      <p className="mt-2 max-w-xl text-muted-foreground">
        {t('home.subtitle')}
      </p>
    </section>
  )
}
