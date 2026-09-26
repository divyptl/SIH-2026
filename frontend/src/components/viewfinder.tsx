import { cn } from 'cn'

/**
 * Corner marks around an image area, like a camera viewfinder's crop marks.
 * Place inside a `relative` container; `inset` pushes them outside its edges.
 */
export function ViewfinderCorners({
  className,
  inset = 0,
}: {
  className?: string
  /** Offset from the container's edges in px; negative sits outside. */
  inset?: number
}) {
  const corner = 'absolute size-3 border-current'
  return (
    <span
      aria-hidden
      className={cn(
        'pointer-events-none absolute text-foreground/35',
        className,
      )}
      style={{ inset }}
    >
      <span className={cn(corner, 'start-0 top-0 border-s-2 border-t-2')} />
      <span className={cn(corner, 'end-0 top-0 border-e-2 border-t-2')} />
      <span className={cn(corner, 'start-0 bottom-0 border-s-2 border-b-2')} />
      <span className={cn(corner, 'end-0 bottom-0 border-e-2 border-b-2')} />
    </span>
  )
}
