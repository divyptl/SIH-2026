import * as React from 'react'
import { RefreshCwIcon } from 'lucide-react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { useTranslation } from 'react-i18next'

import { Button } from '#/components/ui/button'

const PAGE_SIZE = 3
const ROTATE_MS = 7000

/** Fisher-Yates over indices, so every language walks the list in one order. */
function shuffledIndices(length: number) {
  const order = Array.from({ length }, (_, index) => index)
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[order[i], order[j]] = [order[j], order[i]]
  }
  return order
}

interface PromptSuggestionsProps {
  /** Two images switch to change-detection and sensor-comparison questions. */
  imageCount: number
  onPick: (text: string) => void
}

/**
 * Three example questions at a time, cycling through the pregenerated set.
 *
 * Rotation pauses while the pointer or keyboard focus is on the list, while
 * the tab is hidden, and entirely under reduced motion; the refresh button
 * always shows the next three.
 */
export function PromptSuggestions({
  imageCount,
  onPick,
}: PromptSuggestionsProps) {
  const { t } = useTranslation()
  const reduceMotion = useReducedMotion()
  const kind = imageCount >= 2 ? 'pair' : 'single'
  const items = t(`suggestions.${kind}`, { returnObjects: true })

  // One shuffle per pool, kept across language changes so the visible
  // questions translate in place rather than jumping to different ones.
  const [orders] = React.useState(() => ({
    single: shuffledIndices(
      t('suggestions.single', { returnObjects: true }).length,
    ),
    pair: shuffledIndices(
      t('suggestions.pair', { returnObjects: true }).length,
    ),
  }))
  const order = orders[kind]
  const pages = Math.ceil(order.length / PAGE_SIZE)

  const [page, setPage] = React.useState(0)
  const [paused, setPaused] = React.useState(false)
  // Start from the top of the new pool when the image count flips it.
  React.useEffect(() => {
    setPage(0)
  }, [kind])

  React.useEffect(() => {
    if (reduceMotion || paused) return
    const timer = setInterval(() => {
      if (!document.hidden) setPage((current) => (current + 1) % pages)
    }, ROTATE_MS)
    return () => clearInterval(timer)
  }, [reduceMotion, paused, pages, page])

  const visible = order
    .slice(page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE)
    .map((index) => items[index])
    .filter(Boolean)

  return (
    <div
      className="flex flex-col gap-2"
      onPointerEnter={() => setPaused(true)}
      onPointerLeave={() => setPaused(false)}
      onFocus={() => setPaused(true)}
      onBlur={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget)) setPaused(false)
      }}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-muted-foreground">
          {t('suggestions.label')}
        </span>
        <Button
          type="button"
          variant="ghost"
          size="icon-xs"
          aria-label={t('suggestions.more')}
          title={t('suggestions.more')}
          onClick={() => setPage((current) => (current + 1) % pages)}
        >
          <RefreshCwIcon />
        </Button>
      </div>
      {/* Reserved height: rotating never pushes the form below it around. */}
      <div className="min-h-[5.75rem]" aria-live="off">
        <AnimatePresence mode="wait" initial={false}>
          <motion.ul
            key={`${kind}-${page}`}
            className="flex flex-col items-start gap-1.5"
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
          >
            {visible.map((text) => (
              <li key={text} className="max-w-full">
                <Button
                  type="button"
                  variant="outline"
                  size="xs"
                  dir="auto"
                  onClick={() => onPick(text)}
                  className="h-auto min-h-6 max-w-full py-1 text-start whitespace-normal"
                >
                  {text}
                </Button>
              </li>
            ))}
          </motion.ul>
        </AnimatePresence>
      </div>
    </div>
  )
}
