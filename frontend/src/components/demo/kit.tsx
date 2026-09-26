/**
 * Building blocks shared by the demo walkthroughs: animation helpers, the
 * image frame, and the steps every input goes through (ask, check the images,
 * pick the task, answer).
 */
import * as React from 'react'
import {
  ArrowDownIcon,
  ArrowRightIcon,
  CheckIcon,
  CpuIcon,
  LanguagesIcon,
} from 'lucide-react'
import { motion, useReducedMotion } from 'motion/react'
import { cn } from 'cn'

import { ViewfinderCorners } from '#/components/viewfinder'
import { Badge } from '#/components/ui/badge'
import { LANGUAGES } from '#/lib/languages'
import type { AnalysisResponse, ImageInfo, Task, TraceStep } from '#/lib/api'

export const EASE = [0.22, 1, 0.36, 1] as const

/** A recorded /api/analyse response plus the question exactly as it was sent. */
export type RecordedRun = AnalysisResponse & { query: string }

export function formatSeconds(ms: number | undefined) {
  if (ms == null) return ''
  if (ms < 1) return '<1 ms'
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`
}

/** The recorded trace step for a stage (and tool, where a stage repeats). */
export function traceStep(
  run: AnalysisResponse,
  stage: TraceStep['stage'],
  tool?: string,
): TraceStep | undefined {
  return run.trace.steps.find(
    (step) => step.stage === stage && (!tool || step.tool === tool),
  )
}

// --- Animation helpers --------------------------------------------------------

/** Reveals `text` one grapheme at a time, so Indic clusters never split. */
export function useTypewriter(text: string, speed: number, delay: number) {
  const reduceMotion = useReducedMotion()
  const graphemes = React.useMemo(
    () =>
      Array.from(
        new Intl.Segmenter(undefined, { granularity: 'grapheme' }).segment(
          text,
        ),
        (part) => part.segment,
      ),
    [text],
  )
  const [count, setCount] = React.useState(reduceMotion ? graphemes.length : 0)

  React.useEffect(() => {
    if (reduceMotion) return
    let shown = 0
    let timer: ReturnType<typeof setTimeout>
    const tick = () => {
      shown += 1
      setCount(shown)
      if (shown < graphemes.length) timer = setTimeout(tick, speed)
    }
    timer = setTimeout(tick, delay)
    return () => clearTimeout(timer)
  }, [graphemes, speed, delay, reduceMotion])

  return {
    text: graphemes.slice(0, count).join(''),
    done: count >= graphemes.length,
  }
}

/** Counts 0..steps-1 at a fixed pace, then stays on the last value. */
export function useTicker(steps: number, interval: number, delay = 0) {
  const reduceMotion = useReducedMotion()
  const [value, setValue] = React.useState(reduceMotion ? steps - 1 : -1)
  React.useEffect(() => {
    if (reduceMotion) return
    let current = -1
    let timer: ReturnType<typeof setTimeout>
    const tick = () => {
      current += 1
      setValue(current)
      if (current < steps - 1) timer = setTimeout(tick, interval)
    }
    timer = setTimeout(tick, delay)
    return () => clearTimeout(timer)
  }, [steps, interval, delay, reduceMotion])
  return value
}

/** Becomes true after `ms`, or at once when motion is reduced. */
export function useAfter(ms: number) {
  const reduceMotion = useReducedMotion()
  const [done, setDone] = React.useState(Boolean(reduceMotion))
  React.useEffect(() => {
    if (reduceMotion) return
    const timer = setTimeout(() => setDone(true), ms)
    return () => clearTimeout(timer)
  }, [ms, reduceMotion])
  return done
}

export function Appear({
  delay = 0,
  className,
  children,
  y = 10,
}: {
  delay?: number
  className?: string
  children: React.ReactNode
  y?: number
}) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5, ease: EASE }}
    >
      {children}
    </motion.div>
  )
}

export function Chip({
  children,
  delay,
  tone = 'plain',
}: {
  children: React.ReactNode
  delay: number
  tone?: 'plain' | 'strong'
}) {
  return (
    <motion.span
      initial={{ opacity: 0, scale: 0.85 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay, duration: 0.3, ease: EASE }}
      className={cn(
        'inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-medium tabular-nums',
        tone === 'strong'
          ? 'bg-foreground text-background'
          : 'bg-background/85 text-foreground ring-1 ring-foreground/15 backdrop-blur',
      )}
    >
      {children}
    </motion.span>
  )
}

export function CheckBadge({ className }: { className?: string }) {
  return (
    <motion.span
      initial={{ scale: 0 }}
      animate={{ scale: 1 }}
      className={cn(
        'flex size-6 shrink-0 items-center justify-center rounded-full bg-foreground text-background',
        className,
      )}
    >
      <CheckIcon className="size-3.5" />
    </motion.span>
  )
}

// --- Image frame ----------------------------------------------------------------

/** A square image area with viewfinder corners; children draw over it. */
export function Frame({
  src,
  alt,
  children,
  size = 'lg',
  className,
}: {
  src: string
  alt: string
  children?: React.ReactNode
  /** Capped by the viewport height so a step always fits on one screen. */
  size?: 'lg' | 'md'
  className?: string
}) {
  return (
    <div
      className={cn(
        'relative mx-auto w-full',
        size === 'lg'
          ? 'lg:max-w-[calc(100dvh-25rem)]'
          : 'lg:max-w-[calc(100dvh-32rem)]',
        className,
      )}
    >
      <div className="relative aspect-square overflow-hidden rounded-lg ring-1 ring-foreground/10">
        <img src={src} alt={alt} className="size-full object-cover" />
        {children}
      </div>
      <ViewfinderCorners inset={-6} />
    </div>
  )
}

/** A PNG mask (opaque where true) drawn as a coloured layer over an image. */
export function MaskLayer({
  src,
  opacity = 0.6,
  className = 'bg-fuchsia-500',
}: {
  src: string
  opacity?: number
  className?: string
}) {
  const url = `url("${src}")`
  return (
    <div
      aria-hidden
      className={cn('absolute inset-0', className)}
      style={{
        opacity,
        maskImage: url,
        WebkitMaskImage: url,
        maskSize: '100% 100%',
        WebkitMaskSize: '100% 100%',
      }}
    />
  )
}

/** A normalised box drawn over a Frame. */
export function BoxOutline({
  box,
  className,
  children,
}: {
  box: { x_min: number; y_min: number; x_max: number; y_max: number }
  className?: string
  children?: React.ReactNode
}) {
  return (
    <div
      className={cn('absolute', className)}
      style={{
        left: `${box.x_min * 100}%`,
        top: `${box.y_min * 100}%`,
        width: `${(box.x_max - box.x_min) * 100}%`,
        height: `${(box.y_max - box.y_min) * 100}%`,
      }}
    >
      {children}
    </div>
  )
}

// --- Step: ask -------------------------------------------------------------------

/**
 * The question as asked. A question in an Indian language is shown being
 * translated to English; an English one goes straight to the models.
 */
export function AskStep({ run }: { run: RecordedRun }) {
  const translation = run.translation
  const translated =
    translation != null &&
    translation.original_query !== translation.english_query
  const original = run.query
  const typed = useTypewriter(original, translated ? 85 : 45, 400)
  const reduceMotion = useReducedMotion()
  const afterTyping = reduceMotion
    ? 0
    : 0.4 + (original.length * (translated ? 85 : 45)) / 1000 + 0.4
  const native = LANGUAGES.filter((language) => language.code !== 'en')
  const sourceLanguage = LANGUAGES.find(
    (language) => language.code === translation?.source_language,
  )

  return (
    <div className="flex h-full flex-col justify-center gap-6">
      <div className="rounded-2xl bg-card p-6 ring-1 ring-foreground/10">
        <span className="text-xs text-muted-foreground">
          {sourceLanguage?.name ?? 'English'}
        </span>
        <p
          lang={translation?.source_language}
          className="mt-1 min-h-[1.4em] text-3xl font-medium sm:text-4xl"
        >
          {typed.text}
          {!typed.done && (
            <span className="ms-0.5 inline-block h-[1em] w-0.5 translate-y-1 animate-pulse bg-foreground" />
          )}
        </p>
      </div>

      {translated ? (
        <>
          <Appear delay={afterTyping} className="flex items-center gap-3 ps-6">
            <ArrowDownIcon className="size-5 text-muted-foreground" />
            <Badge variant="outline" className="gap-1.5">
              <LanguagesIcon />
              {translation.engine}
            </Badge>
            <span className="text-sm text-muted-foreground tabular-nums">
              {formatSeconds(
                traceStep(run, 'translate', 'query-translator')?.duration_ms,
              )}
            </span>
          </Appear>
          <Appear
            delay={afterTyping + 0.35}
            className="rounded-2xl bg-card p-6 ring-1 ring-foreground/10"
          >
            <span className="text-xs text-muted-foreground">
              English, for the models
            </span>
            <p className="mt-1 font-heading text-2xl font-medium sm:text-3xl">
              {translation.english_query}
            </p>
          </Appear>
        </>
      ) : (
        <Appear
          delay={afterTyping}
          className="flex items-center gap-3 ps-6 text-base text-muted-foreground"
        >
          <ArrowDownIcon className="size-5" />
          Already in English, so it goes straight to the models.
        </Appear>
      )}

      <Appear delay={afterTyping + 0.9} className="overflow-hidden">
        <p className="mb-2 text-sm text-muted-foreground">
          Questions can be asked in any of the {native.length} Indian languages
          the app supports:
        </p>
        <div className="flex flex-wrap gap-1.5">
          {native.map((language, index) => (
            <Chip key={language.code} delay={afterTyping + 1 + index * 0.04}>
              <span lang={language.code}>{language.nativeName}</span>
            </Chip>
          ))}
        </div>
      </Appear>
    </div>
  )
}

// --- Step: check the images ------------------------------------------------------

export interface Shot {
  src: string
  info: ImageInfo
  /** What the image is, e.g. "Dry season" or "Radar (Sentinel-1)". */
  title: string
  /** Right-hand caption, e.g. a date. */
  caption?: string
  modality: 'Optical' | 'Radar (SAR)'
}

export function InputsStep({
  shots,
  verdict,
}: {
  shots: Array<Shot>
  verdict: React.ReactNode
}) {
  const single = shots.length === 1
  return (
    <div className="flex h-full flex-col justify-center gap-8">
      <div className={single ? 'grid' : 'grid grid-cols-2 gap-6'}>
        {shots.map((shot, index) => (
          <motion.div
            key={shot.src}
            initial={{ opacity: 0, y: 30, rotate: index === 0 ? -3 : 3 }}
            animate={{ opacity: 1, y: 0, rotate: 0 }}
            transition={{
              delay: 0.15 + index * 0.25,
              duration: 0.7,
              ease: EASE,
            }}
          >
            <Frame src={shot.src} alt={shot.title} size="md">
              <div className="absolute inset-x-0 bottom-0 flex flex-wrap gap-1.5 p-3">
                <Chip delay={1.1 + index * 0.1} tone="strong">
                  {shot.modality}
                </Chip>
                <Chip delay={1.3 + index * 0.1}>
                  {shot.info.width} × {shot.info.height}
                </Chip>
                {shot.info.ground_sample_distance_m != null && (
                  <Chip delay={1.5 + index * 0.1}>
                    {Number(shot.info.ground_sample_distance_m.toFixed(1))} m
                    per pixel
                  </Chip>
                )}
                <Chip delay={1.7 + index * 0.1}>
                  {shot.info.is_georeferenced
                    ? 'Georeferenced'
                    : 'Not georeferenced'}
                </Chip>
              </div>
            </Frame>
            <p className="mx-auto mt-3 flex w-full items-baseline justify-between gap-2 text-sm lg:max-w-[calc(100dvh-32rem)]">
              <span className="font-medium">{shot.title}</span>
              {shot.caption && (
                <span className="text-muted-foreground tabular-nums">
                  {shot.caption}
                </span>
              )}
            </p>
          </motion.div>
        ))}
      </div>
      <Appear
        delay={2.4}
        className="flex items-center gap-3 rounded-xl bg-card px-5 py-4 ring-1 ring-foreground/10"
      >
        <span className="flex size-7 shrink-0 items-center justify-center rounded-full bg-foreground text-background">
          <CheckIcon className="size-4" />
        </span>
        <p className="text-base">{verdict}</p>
      </Appear>
    </div>
  )
}

// --- Step: pick the task --------------------------------------------------------

/** The controller's routing rules, in the order it applies them. */
const ROUTES: Array<{ input: string; task: Task; name: string }> = [
  {
    input: 'One image, asking where something is',
    task: 'grounding',
    name: 'Find it on the image',
  },
  { input: 'One image, any other question', task: 'vqa', name: 'Answer it' },
  {
    input: 'Two dates of one place',
    task: 'change_vqa',
    name: 'Question about change',
  },
  {
    input: 'Optical + radar of one place',
    task: 'fusion',
    name: 'Optical–radar fusion',
  },
]

export function RouteStep({ run }: { run: RecordedRun }) {
  const chosen = ROUTES.findIndex((route) => route.task === run.task)
  // Rows appear, then the rule for this input lights up.
  const tick = useTicker(ROUTES.length + 1, 400, 300)
  const settled = tick === ROUTES.length
  const english = run.translation?.english_query ?? run.query
  // The word that sent a single image to grounding, quoted in the trace.
  const keyword = traceStep(run, 'classify')?.detail.match(/\("(.+?)"\)/)?.[1]

  return (
    <div className="flex h-full flex-col justify-center gap-6">
      {keyword && (
        <Appear className="rounded-xl bg-card px-5 py-4 text-lg ring-1 ring-foreground/10">
          <Highlight text={english} word={keyword} />
        </Appear>
      )}
      <div className="flex flex-col gap-2.5">
        {ROUTES.map((route, index) => {
          const isChosen = settled && index === chosen
          return (
            <motion.div
              key={route.task}
              initial={{ opacity: 0, x: -16 }}
              animate={{
                opacity: tick < index ? 0 : settled && !isChosen ? 0.4 : 1,
                x: tick < index ? -16 : 0,
              }}
              transition={{ duration: 0.4, ease: EASE }}
              className={cn(
                'grid grid-cols-[minmax(0,1.2fr)_auto_minmax(0,1fr)] items-center gap-4 rounded-xl px-4 py-3.5 ring-1 transition-colors duration-300',
                isChosen
                  ? 'bg-foreground text-background ring-foreground'
                  : 'bg-card ring-foreground/10',
              )}
            >
              <span className="font-heading text-base font-semibold sm:text-lg">
                {route.input}
              </span>
              <ArrowRightIcon
                className={cn(
                  'size-5',
                  isChosen ? 'text-background/70' : 'text-muted-foreground',
                )}
              />
              <span className="flex items-center gap-2 text-base">
                {route.name}
                {isChosen && (
                  <CheckBadge className="ms-auto bg-background text-foreground" />
                )}
              </span>
            </motion.div>
          )
        })}
      </div>
      {settled && (
        <Appear className="flex flex-wrap items-baseline gap-x-6 gap-y-2 rounded-xl bg-card p-5 ring-1 ring-foreground/10">
          <p className="text-lg/7">
            Plain rules, no model: the same input always gets the same task.
          </p>
          <p className="text-sm text-muted-foreground tabular-nums">
            Decided in {formatSeconds(traceStep(run, 'classify')?.duration_ms)}
          </p>
        </Appear>
      )}
    </div>
  )
}

/** `text` with the first whole-word `word` marked. */
function Highlight({ text, word }: { text: string; word: string }) {
  const at = text.toLowerCase().search(new RegExp(`\\b${word}\\b`))
  if (at < 0) return <>{text}</>
  return (
    <>
      {text.slice(0, at)}
      <motion.mark
        initial={{ backgroundColor: 'rgb(217 70 239 / 0)' }}
        animate={{ backgroundColor: 'rgb(217 70 239 / 0.85)' }}
        transition={{ delay: 0.6, duration: 0.4 }}
        className="rounded px-1 text-white"
      >
        {text.slice(at, at + word.length)}
      </motion.mark>
      {text.slice(at + word.length)}
    </>
  )
}

// --- Step: answer ----------------------------------------------------------------

const STAGE_NAMES: Partial<Record<string, string>> = {
  'query-translator': 'Translate question',
  'answer-translator': 'Translate answer',
  validate: 'Check images',
  classify: 'Pick task',
  select: 'Choose model',
  execute: 'Run the model',
  aggregate: 'Combine',
}

export function ConfidenceRow({ run }: { run: AnalysisResponse }) {
  const confidence = Math.round(run.confidence * 100)
  return (
    <div className="flex flex-wrap items-center gap-3">
      <Badge>
        <CpuIcon />
        Fine-tuned model
      </Badge>
      <div className="flex flex-1 items-center gap-2">
        <span className="text-sm text-muted-foreground">Confidence</span>
        <div className="h-1.5 min-w-16 flex-1 overflow-hidden rounded-full bg-muted">
          <motion.div
            className="h-full rounded-full bg-foreground"
            initial={{ width: 0 }}
            animate={{ width: `${confidence}%` }}
            transition={{ delay: 0.9, duration: 1, ease: EASE }}
          />
        </div>
        <span className="text-sm font-semibold tabular-nums">
          {confidence}%
        </span>
      </div>
    </div>
  )
}

/** Every recorded step with its duration, the model's own step in fuchsia. */
export function StepTimeline({ run }: { run: AnalysisResponse }) {
  const steps = run.trace.steps
  const longest = Math.max(...steps.map((step) => step.duration_ms))
  return (
    <div className="rounded-xl bg-card p-4 ring-1 ring-foreground/10">
      <p className="flex items-baseline justify-between text-sm">
        <span className="font-medium">Every step, as recorded</span>
        <span className="text-muted-foreground tabular-nums">
          {formatSeconds(run.execution_time_ms)} in total
        </span>
      </p>
      <ol className="mt-3 flex flex-col gap-1.5">
        {steps.map((step, index) => (
          <li
            key={index}
            className="grid grid-cols-[8.5rem_minmax(0,1fr)_3.5rem] items-center gap-3 text-xs"
          >
            <span className="truncate">
              {STAGE_NAMES[step.tool] ?? STAGE_NAMES[step.stage] ?? step.stage}
            </span>
            <span className="h-1.5 overflow-hidden rounded-full bg-muted">
              <motion.span
                className={cn(
                  'block h-full rounded-full',
                  step.stage === 'execute'
                    ? 'bg-fuchsia-500'
                    : 'bg-foreground/60',
                )}
                initial={{ width: 0 }}
                animate={{
                  width: `${Math.max(1.5, (step.duration_ms / longest) * 100)}%`,
                }}
                transition={{
                  delay: 1.1 + index * 0.08,
                  duration: 0.6,
                  ease: EASE,
                }}
              />
            </span>
            <span className="text-end text-muted-foreground tabular-nums">
              {formatSeconds(step.duration_ms)}
            </span>
          </li>
        ))}
      </ol>
    </div>
  )
}
