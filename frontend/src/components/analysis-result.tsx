import * as React from 'react'
import { CircleAlertIcon, CloudIcon, CpuIcon, ListTreeIcon } from 'lucide-react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { useTranslation } from 'react-i18next'

import { EvidenceOverlay } from '#/components/evidence-overlay'
import { ReportButton } from '#/components/report-button'
import { Alert, AlertDescription, AlertTitle } from '#/components/ui/alert'
import { Badge } from '#/components/ui/badge'
import { Button } from '#/components/ui/button'
import { Card, CardAction, CardHeader, CardTitle } from '#/components/ui/card'
import { Progress } from '#/components/ui/progress'
import { Separator } from '#/components/ui/separator'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '#/components/ui/sheet'
import { ToggleGroup, ToggleGroupItem } from '#/components/ui/toggle-group'
import type { AnalysisResponse } from '#/lib/api'
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
  const englishQuery =
    translation && translation.original_query !== translation.english_query
      ? translation.english_query
      : null

  return (
    <Card className="gap-6">
      <CardHeader>
        <CardTitle className="text-base">{t('result.title')}</CardTitle>
        <CardAction className="flex items-center gap-2">
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
        </CardAction>
      </CardHeader>

      <div className="flex flex-col gap-5 px-(--card-spacing)">
        {reportError && (
          <Alert variant="destructive">
            <CircleAlertIcon />
            <AlertDescription>{reportError}</AlertDescription>
          </Alert>
        )}
        <AnimatePresence mode="wait" initial={false}>
          <motion.p
            key={lang.code}
            lang={lang.code}
            dir={lang.dir}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="max-w-prose text-base/7 whitespace-pre-line"
          >
            {answer}
          </motion.p>
        </AnimatePresence>

        <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
          <div className="flex items-center gap-3">
            <span className="text-sm text-muted-foreground">
              {t('result.confidence')}
            </span>
            <Progress
              value={barValue}
              aria-label={t('result.confidence')}
              className="w-28 **:data-[slot=progress-indicator]:duration-700 **:data-[slot=progress-indicator]:ease-out"
            />
            <span className="text-sm font-medium tabular-nums">
              {confidencePercent}%
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-1.5">
            <Badge variant="secondary">{t(`task.${result.task}`)}</Badge>
            <Badge variant="outline">
              {t(`configuration.${result.trace.input_configuration}`)}
            </Badge>
            <ModelSourceBadge domainAdapted={result.trace.domain_adapted} />
          </div>
          <span className="ms-auto text-xs text-muted-foreground tabular-nums">
            {formatMs(result.execution_time_ms)}
          </span>
        </div>
      </div>

      {previews.length > 0 && (
        <>
          <Separator />
          <div
            className={
              'px-(--card-spacing) ' +
              (previews.length > 1 ? 'grid gap-4 sm:grid-cols-2' : 'grid')
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
                    labelOf={(item) => labelOf(result.evidence.indexOf(item))}
                    labelLang={lang.code}
                    activeIndex={
                      activeEvidence !== null
                        ? forThisImage.indexOf(result.evidence[activeEvidence])
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
                  <figcaption className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
                    {result.inputs.length > 1 && (
                      <span className="font-medium text-foreground">
                        {t('result.image', { index: info.index + 1 })}
                      </span>
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
                    {info.is_georeferenced && (
                      <span>{t('result.georeferenced')}</span>
                    )}
                  </figcaption>
                </figure>
              )
            })}
          </div>
        </>
      )}

      {result.evidence.length > 0 && (
        <section className="flex flex-col gap-3 px-(--card-spacing)">
          <h3 className="text-sm font-medium">
            {t('result.evidence', { count: result.evidence.length })}
          </h3>
          <ol className="flex flex-col divide-y rounded-lg border">
            {result.evidence.map((item, index) => {
              const isSpatial = item.type === 'bbox' && item.data
              const label = labelOf(index)
              return (
                <motion.li
                  key={index}
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.05 * index, duration: 0.2 }}
                  onMouseEnter={() => setActiveEvidence(index)}
                  onMouseLeave={() => setActiveEvidence(null)}
                  className={
                    'flex items-start gap-3 p-3 transition-colors first:rounded-t-lg last:rounded-b-lg ' +
                    (activeEvidence === index ? 'bg-muted' : '')
                  }
                >
                  <span
                    className={
                      'mt-0.5 flex size-5 shrink-0 items-center justify-center rounded text-[11px] font-medium tabular-nums ' +
                      (isSpatial
                        ? 'bg-primary text-primary-foreground'
                        : 'border text-muted-foreground')
                    }
                  >
                    {index + 1}
                  </span>
                  <div
                    lang={lang.code}
                    dir={lang.dir}
                    className="flex min-w-0 flex-col gap-0.5"
                  >
                    {label && <span className="font-medium">{label}</span>}
                    <span className="text-muted-foreground">
                      {descriptionOf(index)}
                    </span>
                  </div>
                  {item.confidence !== null && (
                    <span className="ms-auto shrink-0 text-xs text-muted-foreground tabular-nums">
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
    </Card>
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
