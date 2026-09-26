import { useTranslation } from 'react-i18next'

type Kind = 'single' | 'pair' | 'fusion'

/** A small drawing of what each kind of upload looks like. */
function Diagram({ kind }: { kind: Kind }) {
  const tile = 'absolute size-12 rounded-md ring-1 ring-foreground/20'
  return (
    <div aria-hidden className="graticule relative h-20 w-28 rounded-lg">
      {kind === 'single' && (
        <span className={`${tile} top-4 left-8 bg-foreground/10`} />
      )}
      {kind === 'pair' && (
        <>
          <span className={`${tile} top-2.5 left-5 bg-foreground/5`} />
          <span className={`${tile} top-5.5 left-11 bg-foreground/15`}>
            <span className="absolute top-3 left-2 size-4 rounded-sm bg-fuchsia-500/70" />
          </span>
        </>
      )}
      {kind === 'fusion' && (
        <>
          <span className={`${tile} top-4 left-4 bg-foreground/10`} />
          <span
            className={`${tile} top-4 left-12 bg-[repeating-linear-gradient(135deg,var(--color-foreground)_0_1px,transparent_1px_4px)] opacity-40`}
          />
        </>
      )}
    </div>
  )
}

/**
 * The result area before the first analysis: what each kind of upload can
 * answer, so a first-time user knows what to bring.
 */
export function ResultEmpty() {
  const { t } = useTranslation()
  const kinds: Array<Kind> = ['single', 'pair', 'fusion']

  return (
    <section className="flex flex-col gap-6">
      <h2 className="font-heading text-xl font-semibold tracking-tight">
        {t('guide.title')}
      </h2>
      <ul className="grid gap-8 sm:grid-cols-3 sm:gap-0 sm:divide-x">
        {kinds.map((kind) => (
          <li
            key={kind}
            className="flex flex-col gap-3 sm:px-6 sm:first:ps-0 sm:last:pe-0"
          >
            <Diagram kind={kind} />
            <h3 className="font-heading text-base font-semibold">
              {t(`guide.${kind}.title`)}
            </h3>
            <p className="text-sm/6 text-muted-foreground">
              {t(`guide.${kind}.body`)}
            </p>
          </li>
        ))}
      </ul>
    </section>
  )
}
