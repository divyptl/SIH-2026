import * as React from 'react'
import {
  ChevronsLeftRightIcon,
  CircleAlertIcon,
  CloudIcon,
  Columns2Icon,
  CpuIcon,
  ListTreeIcon,
  MessageSquareTextIcon,
} from 'lucide-react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { useTranslation } from 'react-i18next'

import { ChangeSwipe } from '#/components/change-swipe'
import { EvidenceOverlay } from '#/components/evidence-overlay'
import { ReportButton } from '#/components/report-button'
import { Alert, AlertDescription, AlertTitle } from '#/components/ui/alert'
import { Badge } from '#/components/ui/badge'
import { Button } from '#/components/ui/button'
import { Progress } from '#/components/ui/progress'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '#/components/ui/sheet'
import { ToggleGroup, ToggleGroupItem } from '#/components/ui/toggle-group'
import type { AnalysisResponse, ImageInfo } from '#/lib/api'
import { getLanguage } from '#/lib/languages'
import type { Language } from '#/lib/languages'

interface AnalysisResultProps {
  result: AnalysisResponse
  /** The question as the user submitted it. */
  query: string
}

function formatMs(ms: number) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`
}

/** Ground sample distance with precision that suits its scale: 0.31, 2.5, 10. */
function formatGsd(metres: number) {
  return metres < 1
    ? metres.toFixed(2)
    : metres < 10
      ? metres.toFixed(1)
      : Math.round(metres).toString()
}

const ENGLISH: Pick<Language, 'code' | 'dir'> = { code: 'en', dir: 'ltr' }

type CompareView = 'swipe' | 'side'

/** One image's modality, size, resolution and georeferencing. */
function ImageCaption({
  info,
  heading,
}: {
  info: ImageInfo
  heading: string | null
}) {
  const { t } = useTranslation()
  return (
    // content-start: in a two-column caption row the shorter caption would
    // otherwise be centred against the taller one.
    <span className="flex flex-wrap content-start items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
      {heading && (
        <span className="font-medium text-foreground">{heading}</span>
      )}
      <span>{t(`modality.${info.modality}`)}</span>
      <span className="tabular-nums">
        {info.width}&times;{info.height}
      </span>
      {info.ground_sample_distance_m != null && (
        <span className="tabular-nums">
          {t('result.gsd', {
            value: formatGsd(info.ground_sample_distance_m),
          })}
        </span>
      )}
      {info.is_georeferenced && <span>{t('result.georeferenced')}</span>}
    </span>
  )
}

export function AnalysisResult({ result, query }: AnalysisResultProps) {
  const { t } = useTranslation()
  const reduceMotion = useReducedMotion()
  const [activeEvidence, setActiveEvidence] = React.useState<number | null>(
    null,
  )

  // The models answer in English; the translation layer adds a localised copy
  // of everything the user reads. One toggle flips the whole result between it
  // and the English original.
  const translation = result.translation
  const localLanguage =
    translation?.answer != null
      ? getLanguage(translation.target_language)
      : null
  const [showEnglish, setShowEnglish] = React.useState(false)
  const [view, setView] = React.useState<CompareView>('swipe')
  const [maskOpacity, setMaskOpacity] = React.useState(45)
  const [reportError, setReportError] = React.useState<string | null>(null)
  const localised = localLanguage !== null && !showEnglish
  const lang = localised ? localLanguage : ENGLISH

  const answer = localised ? translation!.answer! : result.answer
  const labelOf = (index: number) =>
    (localised ? translation?.evidence_labels?.[index] : null) ??
    result.evidence[index].label
  const descriptionOf = (index: number) =>
    (localised ? translation?.evidence_descriptions?.[index] : null) ??
    result.evidence[index].description
  const warnings =
    (localised ? translation?.warnings : null) ?? result.trace.warnings
  const rationale =
    (localised ? translation?.routing_rationale : null) ??
    result.trace.routing_rationale

  const confidencePercent = Math.round(result.confidence * 100)
  // The bar fills from empty once mounted; the number itself stays still so
  // nothing around it shifts.
  const [barValue, setBarValue] = React.useState(
    reduceMotion ? confidencePercent : 0,
  )
  React.useEffect(() => {
    const frame = requestAnimationFrame(() => setBarValue(confidencePercent))
    return () => cancelAnimationFrame(frame)
  }, [confidencePercent])
  const observations = result.evidence.filter(
    (item) => item.type === 'observation',
  )
  // The server renders every input to a browser-displayable JPEG; anything
  // without one cannot be shown (and never has boxes drawn on it).
  const previews = result.inputs.filter((info) => info.preview_data_uri)
  // Two dates of one place can be swiped between; other pairs stay side by side.
  const isPair =
    previews.length === 2 &&
    result.trace.input_configuration === 'bi_temporal_pair'
  const hasMask = result.evidence.some(
    (item) => item.type === 'mask' && item.data,
  )
  const englishQuery =
    translation && translation.original_query !== translation.english_query
      ? translation.english_query
      : null

  const viewerControls = (isPair || hasMask) && (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
      {isPair && (
        <ToggleGroup
          variant="outline"
          size="sm"
          aria-label={t('result.view.label')}
          value={[view]}
          onValueChange={(value: Array<string>) => {
            if (value.length > 0) setView(value[0] as CompareView)
          }}
        >
          <ToggleGroupItem value="swipe" className="px-2.5">
            <ChevronsLeftRightIcon />
            {t('result.view.swipe')}
          </ToggleGroupItem>
          <ToggleGroupItem value="side" className="px-2.5">
            <Columns2Icon />
            {t('result.view.side')}
          </ToggleGroupItem>
        </ToggleGroup>
      )}
      {hasMask && (
        <label className="ms-auto flex items-center gap-2 text-xs text-muted-foreground">
          <span className="size-2.5 rounded-full bg-fuchsia-500" />
          {t('result.changeMask')}
          <input
            type="range"
            min={0}
            max={100}
            step={5}
            value={maskOpacity}
            onChange={(event) => setMaskOpacity(Number(event.target.value))}
            aria-label={t('result.maskOpacity')}
            className="w-28 accent-fuchsia-500"
          />
          <span className="w-9 text-end text-foreground tabular-nums">
            {maskOpacity}%
          </span>
        </label>
      )}
    </div>
  )

  return (
    <article className="overflow-hidden rounded-xl bg-card ring-1 ring-foreground/10">
      <header className="flex flex-wrap items-center gap-2 border-b px-4 py-3 sm:px-5">
        <h2 className="me-2 font-heading text-base font-semibold">
          {t('result.title')}
        </h2>
        <Badge variant="secondary">{t(`task.${result.task}`)}</Badge>
        <Badge variant="outline">
          {t(`configuration.${result.trace.input_configuration}`)}
        </Badge>
        <ModelSourceBadge domainAdapted={result.trace.domain_adapted} />
        <div className="ms-auto flex items-center gap-2">
          {localLanguage && (
            <ToggleGroup
              variant="outline"
              size="sm"
              value={[showEnglish ? 'en' : 'local']}
              onValueChange={(value: Array<string>) => {
                if (value.length > 0) setShowEnglish(value[0] === 'en')
              }}
            >
              <ToggleGroupItem
                value="local"
                lang={localLanguage.code}
                className="px-2.5"
              >
                {localLanguage.nativeName}
              </ToggleGroupItem>
              <ToggleGroupItem value="en" lang="en" className="px-2.5">
                English
              </ToggleGroupItem>
            </ToggleGroup>
          )}
          <ReportButton
            result={result}
            query={query}
            language={lang.code}
            onError={setReportError}
          />
          <TraceSheet
            result={result}
            warnings={warnings}
            rationale={rationale}
            lang={lang}
            englishQuery={englishQuery}
          />
        </div>
      </header>

      <div
        className={
          'grid gap-8 p-4 sm:p-5 ' +
          (previews.length > 0
            ? 'lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]'
            : '')
        }
      >
        {previews.length > 0 && (
          <div className="flex min-w-0 flex-col gap-3 lg:sticky lg:top-20 lg:self-start">
            {viewerControls}

            {isPair && view === 'swipe' ? (
              <figure className="flex flex-col gap-2">
                <ChangeSwipe
                  before={{
                    src: previews[0].preview_data_uri!,
                    alt: previews[0].filename,
                    label: t('result.before'),
                  }}
                  after={{
                    src: previews[1].preview_data_uri!,
                    alt: previews[1].filename,
                    label: t('result.after'),
                  }}
                  dividerLabel={t('result.divider')}
                  // The pair shares a footprint, so evidence for either image
                  // marks the same ground.
                  evidence={result.evidence}
                  numberOf={(item) => result.evidence.indexOf(item) + 1}
                  labelOf={(item) => labelOf(result.evidence.indexOf(item))}
                  labelLang={lang.code}
                  maskOpacity={maskOpacity / 100}
                  activeIndex={activeEvidence}
                />
                <figcaption className="grid gap-x-4 gap-y-1 sm:grid-cols-2">
                  <ImageCaption
                    info={previews[0]}
                    heading={t('result.before')}
                  />
                  <ImageCaption
                    info={previews[1]}
                    heading={t('result.after')}
                  />
                </figcaption>
              </figure>
            ) : (
              <div
                className={
                  previews.length > 1 ? 'grid gap-4 sm:grid-cols-2' : 'grid'
                }
              >
                {previews.map((info) => {
                  const forThisImage = result.evidence.filter(
                    (item) => item.image_index === info.index,
                  )
                  return (
                    <figure key={info.index} className="flex flex-col gap-2">
                      <EvidenceOverlay
                        src={info.preview_data_uri!}
                        alt={info.filename}
                        evidence={forThisImage}
                        numberOf={(item) => result.evidence.indexOf(item) + 1}
                        labelOf={(item) =>
                          labelOf(result.evidence.indexOf(item))
                        }
                        labelLang={lang.code}
                        maskOpacity={maskOpacity / 100}
                        activeIndex={
                          activeEvidence !== null
                            ? forThisImage.indexOf(
                                result.evidence[activeEvidence],
                              )
                            : null
                        }
                        onHoverChange={(local) =>
                          setActiveEvidence(
                            local === null
                              ? null
                              : result.evidence.indexOf(forThisImage[local]),
                          )
                        }
                      />
                      <figcaption>
                        <ImageCaption
                          info={info}
                          heading={
                            result.inputs.length > 1
                              ? t('result.image', { index: info.index + 1 })
                              : null
                          }
                        />
                      </figcaption>
                    </figure>
                  )
                })}
              </div>
            )}
          </div>
        )}

        <div className="flex min-w-0 flex-col gap-6">
          {reportError && (
            <Alert variant="destructive">
              <CircleAlertIcon />
              <AlertDescription>{reportError}</AlertDescription>
            </Alert>
          )}

          <div className="flex flex-col gap-3">
            <AnimatePresence mode="wait" initial={false}>
              <motion.p
                key={lang.code}
                lang={lang.code}
                dir={lang.dir}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.15 }}
                className="max-w-prose font-heading text-xl/8 font-medium tracking-[-0.01em] text-pretty whitespace-pre-line"
              >
                {answer}
              </motion.p>
            </AnimatePresence>
            {result.narration && (
              <p className="flex max-w-prose items-start gap-1.5 text-xs/5 text-muted-foreground">
                <MessageSquareTextIcon className="mt-0.5 size-3.5 shrink-0" />
                <span>
                  {t('result.narration', { model: result.narration.model })}
                </span>
              </p>
            )}
          </div>

          <div className="flex items-center gap-3 rounded-lg bg-muted/60 px-3 py-2.5">
            <span className="text-sm text-muted-foreground">
              {t('result.confidence')}
            </span>
            <Progress
              value={barValue}
              aria-label={t('result.confidence')}
              className="flex-1 **:data-[slot=progress-indicator]:duration-700 **:data-[slot=progress-indicator]:ease-out"
            />
            <span className="w-10 text-end text-sm font-semibold tabular-nums">
              {confidencePercent}%
            </span>
            <span className="border-s ps-3 text-xs text-muted-foreground tabular-nums">
              {formatMs(result.execution_time_ms)}
            </span>
          </div>

          {result.evidence.length > 0 && (
            <section className="flex flex-col gap-2">
              <h3 className="font-heading text-sm font-semibold">
                {t('result.evidence', { count: result.evidence.length })}
              </h3>
              <ol className="-mx-2 flex flex-col">
                {result.evidence.map((item, index) => {
                  const isSpatial =
                    (item.type === 'bbox' || item.type === 'mask') && item.data
                  const label = labelOf(index)
                  return (
                    <motion.li
                      key={index}
                      initial={{ opacity: 0, y: 4 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.04 * index, duration: 0.2 }}
                      onMouseEnter={() => setActiveEvidence(index)}
                      onMouseLeave={() => setActiveEvidence(null)}
                      className={
                        'flex items-start gap-3 rounded-lg p-2 transition-colors ' +
                        (activeEvidence === index ? 'bg-muted' : '')
                      }
                    >
                      <span
                        className={
                          'flex size-6 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold tabular-nums ' +
                          (item.type === 'mask'
                            ? 'bg-fuchsia-500 text-white'
                            : isSpatial
                              ? 'bg-foreground text-background'
                              : 'text-muted-foreground ring-1 ring-foreground/20')
                        }
                      >
                        {index + 1}
                      </span>
                      <div
                        lang={lang.code}
                        dir={lang.dir}
                        className="flex min-w-0 flex-1 flex-col gap-0.5 text-sm"
                      >
                        {label && (
                          <span className="font-medium first-letter:uppercase">
                            {label}
                          </span>
                        )}
                        <span className="text-muted-foreground">
                          {descriptionOf(index)}
                        </span>
                      </div>
                      {item.confidence !== null && (
                        <span className="shrink-0 pt-0.5 text-xs text-muted-foreground tabular-nums">
                          {Math.round(item.confidence * 100)}%
                        </span>
                      )}
                    </motion.li>
                  )
                })}
              </ol>
              {observations.length > 0 && previews.length > 0 && (
                <p className="text-xs text-muted-foreground">
                  {t('result.noSpatial', { count: observations.length })}
                </p>
              )}
            </section>
          )}
        </div>
      </div>
    </article>
  )
}

/**
 * Whether a fine-tuned remote-sensing specialist or the generic VLM baseline
 * produced the answer, so a baseline answer is never mistaken for the former.
 */
function ModelSourceBadge({ domainAdapted }: { domainAdapted: boolean }) {
  const { t } = useTranslation()
  const source = domainAdapted ? 'specialist' : 'baseline'
  return (
    <Badge
      variant={domainAdapted ? 'default' : 'outline'}
      title={t(`result.source.${source}Hint`)}
    >
      {domainAdapted ? <CpuIcon /> : <CloudIcon />}
      {t(`result.source.${source}`)}
    </Badge>
  )
}

/** Routing, controller notes and the step-by-step trace, out of the way. */
function TraceSheet({
  result,
  warnings,
  rationale,
  lang,
  englishQuery,
}: {
  result: AnalysisResponse
  warnings: Array<string>
  rationale: string | null
  lang: Pick<Language, 'code' | 'dir'>
  englishQuery: string | null
}) {
  const { t } = useTranslation()
  const title = t('result.trace', { count: result.trace.steps.length })

  return (
    <Sheet>
      <SheetTrigger
        render={
          <Button variant="outline" size="icon-sm" aria-label={title}>
            <ListTreeIcon />
          </Button>
        }
      />
      <SheetContent className="w-full overflow-y-auto sm:max-w-md">
        <SheetHeader>
          <SheetTitle>{title}</SheetTitle>
          <SheetDescription className="font-mono text-xs break-all">
            {result.request_id}
          </SheetDescription>
        </SheetHeader>

        <div className="flex flex-col gap-6 px-4 pb-6">
          {englishQuery && (
            <div className="flex flex-col gap-1 text-sm">
              <span className="text-muted-foreground">
                {t('result.queryTranslated')}
              </span>
              <q lang="en" dir="ltr">
                {englishQuery}
              </q>
            </div>
          )}

          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
            <dt className="text-muted-foreground">{t('result.routing')}</dt>
            <dd>
              {t(`task.${result.trace.task}`)}
              {rationale && (
                <p
                  lang={lang.code}
                  dir={lang.dir}
                  className="mt-1 text-muted-foreground"
                >
                  {rationale}
                </p>
              )}
            </dd>
            <dt className="text-muted-foreground">{t('result.tools')}</dt>
            <dd className="font-mono text-xs leading-5">
              {result.trace.selected_tools.join(', ')}
            </dd>
            <dt className="text-muted-foreground">{t('result.model')}</dt>
            <dd>
              <span className="font-mono text-xs leading-5 break-all">
                {result.model_name}
              </span>
              <p className="mt-1 text-muted-foreground">
                {t(
                  `result.source.${result.trace.domain_adapted ? 'specialist' : 'baseline'}Hint`,
                )}
              </p>
            </dd>
            {result.usage?.total_tokens != null && (
              <>
                <dt className="text-muted-foreground">{t('result.tokens')}</dt>
                <dd className="tabular-nums">{result.usage.total_tokens}</dd>
              </>
            )}
          </dl>

          {warnings.length > 0 && (
            <Alert>
              <CircleAlertIcon />
              <AlertTitle>{t('result.controllerNotes')}</AlertTitle>
              <AlertDescription>
                <ul
                  lang={lang.code}
                  dir={lang.dir}
                  className="ms-4 flex list-disc flex-col gap-1"
                >
                  {warnings.map((warning, index) => (
                    <li key={index}>{warning}</li>
                  ))}
                </ul>
              </AlertDescription>
            </Alert>
          )}

          <ol className="flex flex-col">
            {result.trace.steps.map((step, index) => (
              <li key={index} className="relative flex gap-3 pb-5 last:pb-0">
                <div className="flex flex-col items-center">
                  <span className="mt-1.5 size-2 shrink-0 rounded-full bg-foreground" />
                  {index < result.trace.steps.length - 1 && (
                    <span className="mt-1 w-px flex-1 bg-border" />
                  )}
                </div>
                <div className="flex min-w-0 flex-1 flex-col gap-1 text-sm">
                  <div className="flex items-center gap-2">
                    <span className="font-medium">{step.stage}</span>
                    <span className="truncate font-mono text-xs text-muted-foreground">
                      {step.tool}
                    </span>
                    <span className="ms-auto text-xs text-muted-foreground tabular-nums">
                      {formatMs(step.duration_ms)}
                    </span>
                  </div>
                  {step.detail && (
                    <p className="text-xs/5 text-muted-foreground" dir="auto">
                      {step.detail}
                    </p>
                  )}
                </div>
              </li>
            ))}
          </ol>
        </div>
      </SheetContent>
    </Sheet>
  )
}
