import { cn } from 'cn'

import type { Evidence } from '#/lib/api'

interface EvidenceOverlayProps {
  src: string
  alt: string
  evidence: Array<Evidence>
  /** Index of the evidence item to emphasise, or null for none. */
  activeIndex: number | null
  onHoverChange?: (index: number | null) => void
  className?: string
}

/**
 * Renders an image with the model's bounding boxes drawn over it.
 *
 * Boxes are positioned with percentages so the overlay tracks the rendered
 * size of the image, whatever the original raster dimensions were.
 */
export function EvidenceOverlay({
  src,
  alt,
  evidence,
  activeIndex,
  onHoverChange,
  className,
}: EvidenceOverlayProps) {
  const boxes = evidence.filter((item) => item.type === 'bbox' && item.data)

  return (
    <div
      className={cn(
        'relative overflow-hidden rounded-lg border bg-muted/30',
        className,
      )}
    >
      <img src={src} alt={alt} className="block w-full" />

      {boxes.map((item) => {
        const index = evidence.indexOf(item)
        const box = item.data!
        const isActive = activeIndex === index
        const isDimmed = activeIndex !== null && !isActive

        return (
          <div
            key={index}
            onMouseEnter={() => onHoverChange?.(index)}
            onMouseLeave={() => onHoverChange?.(null)}
            className={cn(
              'absolute rounded-xs border-2 transition-opacity',
              isActive
                ? 'border-primary shadow-[0_0_0_9999px_rgba(0,0,0,0.35)]'
                : 'border-primary/70',
              isDimmed && 'opacity-25',
            )}
            style={{
              left: `${box.x_min * 100}%`,
              top: `${box.y_min * 100}%`,
              width: `${(box.x_max - box.x_min) * 100}%`,
              height: `${(box.y_max - box.y_min) * 100}%`,
            }}
          >
            <span className="absolute -top-px left-0 max-w-full -translate-y-full truncate rounded-t-xs bg-primary px-1 py-px text-[10px] leading-tight font-medium text-primary-foreground">
              {item.label ?? `Region ${index + 1}`}
            </span>
          </div>
        )
      })}
    </div>
  )
}
