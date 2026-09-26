import { useReducedMotion } from 'motion/react'
import { useTranslation } from 'react-i18next'

import { usePreviewUrl } from '#/components/image-slot'
import { ViewfinderCorners } from '#/components/viewfinder'
import { Skeleton } from '#/components/ui/skeleton'
import { Spinner } from '#/components/ui/spinner'

/**
 * Placeholder for the result while the request is in flight: the result's
 * layout, with a line sweeping the image area as the models read it.
 */
export function AnalysisPending({ files }: { files: Array<File> }) {
  const { t } = useTranslation()
  const reduceMotion = useReducedMotion()

  return (
    <div className="grid gap-6 rounded-xl bg-card p-4 ring-1 ring-foreground/10 sm:p-5 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)] lg:gap-8">
      <div className="relative">
        <div className="graticule relative aspect-square overflow-hidden rounded-lg bg-muted ring-1 ring-foreground/10">
          <div
            className={
              files.length > 1
                ? 'grid size-full grid-cols-2 gap-px'
                : 'size-full'
            }
          >
            {files.map((file) => (
              <PendingImage key={file.name + file.size} file={file} />
            ))}
          </div>
          {!reduceMotion && (
            <div aria-hidden className="absolute inset-0 animate-scan">
              <div className="absolute inset-x-0 bottom-0 h-24 bg-gradient-to-b from-transparent to-fuchsia-500/20" />
              <div className="absolute inset-x-0 bottom-0 h-px bg-fuchsia-500 shadow-[0_0_12px_2px] shadow-fuchsia-500/60" />
            </div>
          )}
        </div>
        <ViewfinderCorners inset={-6} />
      </div>

      <div className="flex flex-col gap-4">
        <p className="flex items-center gap-2 font-heading text-lg font-semibold">
          <Spinner />
          {t('home.analyzing')}
        </p>
        <p className="text-sm text-muted-foreground">{t('home.running')}</p>
        <div className="mt-2 flex flex-col gap-2">
          <Skeleton className="h-5 w-full" />
          <Skeleton className="h-5 w-11/12" />
          <Skeleton className="h-5 w-3/5" />
        </div>
        <div className="mt-4 flex flex-col gap-3">
          {[0, 1, 2].map((row) => (
            <div key={row} className="flex items-start gap-3">
              <Skeleton className="size-6 shrink-0 rounded-full" />
              <div className="flex flex-1 flex-col gap-1.5">
                <Skeleton className="h-4 w-2/5" />
                <Skeleton className="h-4 w-4/5" />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

/** A muted preview of an attached image; TIFF has none, so the grid shows. */
function PendingImage({ file }: { file: File }) {
  const url = usePreviewUrl(file)
  if (!url) return <div />
  return (
    <img
      src={url}
      alt=""
      className="size-full object-cover opacity-60 grayscale"
    />
  )
}
