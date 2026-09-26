import type * as React from 'react'
import { cn } from 'cn'
import { useTranslation } from 'react-i18next'

import type { BoundingBox, ChangeMask, Evidence } from '#/lib/api'

export interface EvidenceOverlayProps {
  src: string
  alt: string
  evidence: Array<Evidence>
  /** The number shown on a box, matching its row in the evidence list. */
  numberOf: (item: Evidence) => number
  /** The label shown on a box, in the language the result is displayed in. */
  labelOf?: (item: Evidence) => string | null
  labelLang?: string
  /** Index of the evidence item to emphasise, or null for none. */
  activeIndex: number | null
  onHoverChange?: (index: number | null) => void
  /** Opacity of change masks, 0-1; 0 hides them. Boxes are always drawn. */
  maskOpacity?: number
  /** Layers drawn above the masks and below the boxes. */
  children?: React.ReactNode
  className?: string
}

/**
 * Renders an image with the model's change masks and bounding boxes over it.
 *
 * A mask is a PNG that is opaque where the model predicts change; it is used as
 * a CSS mask over a tinted layer, so the tint follows the design rather than
 * being baked into the image.
 *
 * Boxes are positioned with percentages so the overlay tracks the rendered
 * size of the image, whatever the original raster dimensions were. They are
 * white with a dark outline so they stay legible on both bright optical and
 * dark SAR imagery.
 */
export function EvidenceOverlay({
  src,
  alt,
  evidence,
  numberOf,
  labelOf = (item) => item.label,
  labelLang,
  activeIndex,
  onHoverChange,
  maskOpacity = 0.45,
  children,
  className,
}: EvidenceOverlayProps) {
  const { t } = useTranslation()
  const boxes = evidence.filter((item) => item.type === 'bbox' && item.data)
  const masks =
    maskOpacity > 0
      ? evidence.filter((item) => item.type === 'mask' && item.data)
      : []

  return (
    <div
      className={cn('relative overflow-hidden rounded-lg border', className)}
    >
      <img src={src} alt={alt} className="block w-full" />

      {masks.map((item) => {
        const index = evidence.indexOf(item)
        const url = `url("${(item.data as ChangeMask).png}")`
        return (
          <div
            key={index}
            aria-hidden
            className={cn(
              'pointer-events-none absolute inset-0 animate-in bg-fuchsia-500 duration-300 fade-in',
              'transition-opacity',
            )}
            style={{
              // Hovering the mask's evidence row makes it stand out.
              opacity:
                activeIndex === index
                  ? Math.min(1, maskOpacity + 0.25)
                  : maskOpacity,
              maskImage: url,
              WebkitMaskImage: url,
              maskSize: '100% 100%',
              WebkitMaskSize: '100% 100%',
            }}
          />
        )
      })}

      {children}

      {boxes.map((item) => {
        const index = evidence.indexOf(item)
        const box = item.data as BoundingBox
        const isActive = activeIndex === index
        const isDimmed = activeIndex !== null && !isActive
        const number = numberOf(item)

        return (
          <div
            key={index}
            onMouseEnter={() => onHoverChange?.(index)}
            onMouseLeave={() => onHoverChange?.(null)}
            className={cn(
              'absolute animate-in rounded-[2px] border-2 border-white shadow-[0_0_0_1px_rgb(0_0_0/0.6)] duration-300 fade-in zoom-in-95',
              'transition-[opacity,box-shadow]',
              isActive &&
                'shadow-[0_0_0_1px_rgb(0_0_0/0.6),0_0_0_9999px_rgb(0_0_0/0.45)]',
              isDimmed && 'opacity-30',
            )}
            style={{
              left: `${box.x_min * 100}%`,
              top: `${box.y_min * 100}%`,
              width: `${(box.x_max - box.x_min) * 100}%`,
              height: `${(box.y_max - box.y_min) * 100}%`,
            }}
          >
            <span
              lang={labelLang}
              className="absolute -top-px left-0 flex max-w-[calc(100%+4rem)] -translate-y-full items-center gap-1 truncate rounded-t-[2px] bg-white px-1.5 py-px text-[10px] leading-tight font-medium text-black"
            >
              <span className="tabular-nums">{number}</span>
              {labelOf(item) ?? t('result.region', { index: number })}
            </span>
          </div>
        )
      })}
    </div>
  )
}
