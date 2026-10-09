import * as React from 'react'
import { DownloadIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

import { Button } from '#/components/ui/button'
import { Spinner } from '#/components/ui/spinner'
import { ApiError, downloadReport } from '#/lib/api'
import type { AnalysisResponse, ReportLabels } from '#/lib/api'
import i18n, { loadLocale } from '#/lib/i18n'
import { getLanguage } from '#/lib/languages'

interface ReportButtonProps {
  result: AnalysisResponse
  query: string
  /** Language the report is written in: the one the result is shown in. */
  language: string
  /** Called with a message when the export fails, and null when it retries. */
  onError: (message: string | null) => void
}

/** Dates in the report's language where the browser knows it, else numeric. */
function formatDate(date: Date, language: string) {
  const [locale] = Intl.DateTimeFormat.supportedLocalesOf(language)
  return locale
    ? new Intl.DateTimeFormat(locale, {
        dateStyle: 'long',
        timeStyle: 'short',
      }).format(date)
    : new Intl.DateTimeFormat('en-GB', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      }).format(date)
}

async function buildLabels(
  result: AnalysisResponse,
  language: string,
): Promise<ReportLabels> {
  // The report may be in a language other than the current UI one (the user
  // can switch after the analysis), so load and read that locale explicitly.
  await loadLocale(language)
  const t = i18n.getFixedT(language)
  const observations = result.evidence.filter(
    (item) => item.type === 'observation',
  ).length
  const hasPreviews = result.inputs.some((info) => info.preview_data_uri)

  return {
    title: t('report.title'),
    generated: t('report.generated', {
      date: formatDate(new Date(), language),
    }),
    request: t('result.request'),
    question: t('home.promptLabel'),
    query_translated: t('result.queryTranslated'),
    answer: t('report.answer'),
    english_original: t('result.englishOriginal'),
    task: t('report.task'),
    task_name: t(`task.${result.task}`),
    input: t('report.input'),
    configuration_name: t(`configuration.${result.trace.input_configuration}`),
    confidence: t('result.confidence'),
    time: t('report.time'),
    tokens: t('result.tokens'),
    language: t('report.language'),
    language_name: getLanguage(language).nativeName,
    images: t('home.imagesLabel'),
    // Left as a placeholder; the template fills in each image's number.
    image: t('result.image', { index: '{{index}}' }),
    modality: {
      optical: t('modality.optical'),
      sar: t('modality.sar'),
      unknown: t('modality.unknown'),
    },
    georeferenced: t('result.georeferenced'),
    evidence: t('result.evidence', { count: result.evidence.length }),
    region: t('result.region', { index: '{{index}}' }),
    no_spatial:
      observations > 0 && hasPreviews
        ? t('result.noSpatial', { count: observations })
        : null,
    controller_notes: t('result.controllerNotes'),
    trace: t('result.trace', { count: result.trace.steps.length }),
    routing: t('result.routing'),
    tools: t('result.tools'),
  }
}

/** Downloads the result as a typeset PDF report. */
export function ReportButton({
  result,
  query,
  language,
  onError,
}: ReportButtonProps) {
  const { t } = useTranslation()
  const [isPending, setIsPending] = React.useState(false)

  const download = async () => {
    setIsPending(true)
    onError(null)
    try {
      const pdf = await downloadReport({
        result,
        query,
        language,
        labels: await buildLabels(result, language),
      })
      const url = URL.createObjectURL(pdf)
      const link = document.createElement('a')
      link.href = url
      link.download = `satquery-report-${result.request_id.slice(0, 8)}.pdf`
      link.click()
      // Give the browser a moment to start the download before revoking.
      setTimeout(() => URL.revokeObjectURL(url), 1000)
    } catch (cause) {
      onError(
        cause instanceof ApiError && cause.status === 0
          ? cause.message
          : t('report.failed'),
      )
    } finally {
      setIsPending(false)
    }
  }

  return (
    <Button
      variant="outline"
      size="sm"
      onClick={() => void download()}
      disabled={isPending}
      aria-label={t('report.download')}
    >
      {isPending ? (
        <Spinner data-icon="inline-start" />
      ) : (
        <DownloadIcon data-icon="inline-start" />
      )}
      <span className="max-sm:hidden">{t('report.download')}</span>
    </Button>
  )
}
