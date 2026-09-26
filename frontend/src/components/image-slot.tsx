import * as React from 'react'
import { ImageUpIcon, PlusIcon, SatelliteIcon, XIcon } from 'lucide-react'
import { cn } from 'cn'
import { useTranslation } from 'react-i18next'

import { ViewfinderCorners } from '#/components/viewfinder'
import { Button } from '#/components/ui/button'

export interface DropHandlers {
  onDragEnter: (e: React.DragEvent) => void
  onDragOver: (e: React.DragEvent) => void
  onDragLeave: (e: React.DragEvent) => void
  onDrop: (e: React.DragEvent) => void
}

/** Short format label for an attached file, from its extension. */
export function formatLabel(file: File) {
  const extension = file.name.toLowerCase().split('.').pop() ?? ''
  if (extension === 'png') return 'PNG'
  if (extension === 'jpg' || extension === 'jpeg') return 'JPEG'
  return 'TIFF'
}

export function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

/** A browser-displayable preview URL for PNG/JPEG; TIFF has none. */
export function usePreviewUrl(file: File) {
  const [url, setUrl] = React.useState<string | null>(null)
  React.useEffect(() => {
    if (formatLabel(file) === 'TIFF') return
    const objectUrl = URL.createObjectURL(file)
    setUrl(objectUrl)
    return () => {
      URL.revokeObjectURL(objectUrl)
      setUrl(null)
    }
  }, [file])
  return url
}

/** An attached image: its preview (or a stand-in for TIFF) and file details. */
export function FilledSlot({
  file,
  serverPreview,
  onRemove,
}: {
  file: File
  /** The server's rendering of this file from the last analysis, if any. */
  serverPreview?: string | null
  onRemove: () => void
}) {
  const { t } = useTranslation()
  const preview = usePreviewUrl(file) ?? serverPreview

  return (
    <div className="relative aspect-square overflow-hidden rounded-lg bg-muted ring-1 ring-foreground/10">
      {preview ? (
        <img
          src={preview}
          alt={t('home.previewAlt', { name: file.name })}
          className="size-full object-cover"
        />
      ) : (
        <div className="graticule flex size-full flex-col items-center justify-center gap-2 p-4 text-center text-muted-foreground">
          <SatelliteIcon className="size-6" />
          <span className="max-w-[14rem] text-xs/relaxed">
            {t('home.tiffPreview')}
          </span>
        </div>
      )}
      <div className="absolute inset-x-0 bottom-0 flex items-center gap-2 bg-gradient-to-t from-black/75 to-transparent p-2 pt-8 text-white">
        <div className="flex min-w-0 flex-col">
          <span className="truncate text-xs font-medium" dir="auto">
            {file.name}
          </span>
          <span className="flex gap-2 text-[11px] text-white/70 tabular-nums">
            <span>{formatLabel(file)}</span>
            <span>{formatBytes(file.size)}</span>
          </span>
        </div>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          aria-label={t('home.remove', { name: file.name })}
          className="ms-auto shrink-0 text-white hover:bg-white/15 hover:text-white"
          onClick={onRemove}
        >
          <XIcon />
        </Button>
      </div>
    </div>
  )
}

/** An empty slot: a drop target that also opens the file picker. */
export function EmptySlot({
  primary,
  isDragging,
  dropHandlers,
  onPick,
}: {
  /** The first slot invites the upload; the second offers the optional pair. */
  primary: boolean
  isDragging: boolean
  dropHandlers: DropHandlers
  onPick: () => void
}) {
  const { t } = useTranslation()

  return (
    <button
      type="button"
      {...dropHandlers}
      onClick={onPick}
      className={cn(
        'graticule group/slot relative flex aspect-square w-full flex-col items-center justify-center gap-2 rounded-lg p-4 text-center transition-colors',
        'ring-1 ring-foreground/10 hover:bg-muted/60 focus-visible:ring-3 focus-visible:ring-ring/60 focus-visible:outline-none',
        isDragging && 'bg-muted ring-2 ring-foreground/40',
      )}
    >
      <ViewfinderCorners inset={8} />
      <span
        className={cn(
          'flex size-10 items-center justify-center rounded-full transition-transform group-hover/slot:scale-105',
          primary
            ? 'bg-primary text-primary-foreground'
            : 'bg-background text-muted-foreground ring-1 ring-foreground/10',
        )}
      >
        {primary ? (
          <ImageUpIcon className="size-4" />
        ) : (
          <PlusIcon className="size-4" />
        )}
      </span>
      <span
        className={cn(
          'text-sm font-medium',
          !primary && 'text-muted-foreground',
        )}
      >
        {isDragging
          ? t('home.dropToAttach')
          : primary
            ? t('home.dropHere')
            : t('home.secondSlot')}
      </span>
      <span className="hidden max-w-[15rem] text-xs/relaxed text-muted-foreground sm:block">
        {primary ? t('home.browseHint') : t('home.secondSlotHint')}
      </span>
    </button>
  )
}
