import * as React from 'react'
import { ChevronsLeftRightIcon } from 'lucide-react'

import { EvidenceOverlay } from '#/components/evidence-overlay'
import type { EvidenceOverlayProps } from '#/components/evidence-overlay'

interface SwipeImage {
  src: string
  alt: string
  /** Corner tag, e.g. "Before". */
  label: string
}

interface ChangeSwipeProps extends Omit<
  EvidenceOverlayProps,
  'src' | 'alt' | 'children' | 'onHoverChange'
> {
  before: SwipeImage
  after: SwipeImage
  /** Accessible name of the divider control. */
  dividerLabel: string
}

/**
 * A before/after comparison of a co-registered pair.
 *
 * The later image carries the evidence (change mask and boxes); the earlier
 * one is layered on top and clipped to the left of the divider, so dragging it
 * reveals the change. Boxes stay drawn across both halves: the pair shares one
 * footprint, so a box marks the same ground in either image.
 *
 * The divider is a native range input stretched invisibly over the image,
 * which gives drag, click-to-jump, touch and arrow-key control for free.
 */
export function ChangeSwipe({
  before,
  after,
  dividerLabel,
  ...overlay
}: ChangeSwipeProps) {
  const [position, setPosition] = React.useState(50)

  return (
    // Always left-to-right: the divider tracks the pointer, not the reading order.
    <div dir="ltr" className="group relative touch-none select-none">
      <EvidenceOverlay src={after.src} alt={after.alt} {...overlay}>
        <img
          src={before.src}
          alt={before.alt}
          draggable={false}
          className="absolute inset-0 size-full object-fill"
          style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
        />
        <span className="pointer-events-none absolute bottom-2 left-2 rounded bg-black/65 px-1.5 py-0.5 text-[11px] font-medium text-white">
          {before.label}
        </span>
        <span className="pointer-events-none absolute right-2 bottom-2 rounded bg-black/65 px-1.5 py-0.5 text-[11px] font-medium text-white">
          {after.label}
        </span>
        <div
          aria-hidden
          className="pointer-events-none absolute inset-y-0 z-10 w-0.5 -translate-x-1/2 bg-white shadow-[0_0_0_1px_rgb(0_0_0/0.45)]"
          style={{ left: `${position}%` }}
        >
          <div className="absolute top-1/2 left-1/2 flex size-8 -translate-1/2 items-center justify-center rounded-full border bg-background text-foreground shadow-md group-has-[input:focus-visible]:ring-3 group-has-[input:focus-visible]:ring-ring/60">
            <ChevronsLeftRightIcon className="size-4" />
          </div>
        </div>
      </EvidenceOverlay>
      <input
        type="range"
        min={0}
        max={100}
        step={1}
        value={position}
        onChange={(event) => setPosition(Number(event.target.value))}
        aria-label={dividerLabel}
        aria-valuetext={`${position}%`}
        className="absolute inset-0 z-20 size-full cursor-ew-resize opacity-0"
      />
    </div>
  )
}
