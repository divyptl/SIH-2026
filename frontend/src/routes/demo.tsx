import * as React from 'react'
import { Link, createFileRoute } from '@tanstack/react-router'
import {
  ChevronLeftIcon,
  ChevronRightIcon,
  MaximizeIcon,
  MinimizeIcon,
  PauseIcon,
  PlayIcon,
  RotateCcwIcon,
} from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { cn } from 'cn'

import { changeScenario } from '#/components/demo/change'
import { fusionScenario } from '#/components/demo/fusion'
import { groundingScenario } from '#/components/demo/grounding'
import { formatSeconds } from '#/components/demo/kit'
import type { Scenario } from '#/components/demo/scenario'
import { Badge } from '#/components/ui/badge'
import { Button } from '#/components/ui/button'
import { Kbd, KbdGroup } from '#/components/ui/kbd'
import { Progress } from '#/components/ui/progress'
import { Tabs, TabsList, TabsTrigger } from '#/components/ui/tabs'

export const Route = createFileRoute('/demo')({ component: Demo })

/** One recorded run per input type, each showcasing a fine-tuned model. */
const SCENARIOS: Array<Scenario> = [
  changeScenario,
  groundingScenario,
  fusionScenario,
]

/** Enter or leave full screen, where the browser allows it (not iPhone Safari). */
function toggleFullscreen() {
  if (document.fullscreenElement) {
    void document.exitFullscreen()
  } else if (document.fullscreenEnabled) {
    void document.documentElement.requestFullscreen()
  }
}

/** Whether the page is full screen, including when Esc or the browser left it. */
function useIsFullscreen() {
  const [isFullscreen, setIsFullscreen] = React.useState(false)
  React.useEffect(() => {
    const update = () => setIsFullscreen(document.fullscreenElement !== null)
    document.addEventListener('fullscreenchange', update)
    return () => document.removeEventListener('fullscreenchange', update)
  }, [])
  return isFullscreen
}

interface Position {
  scenario: number
  step: number
}

function Demo() {
  const [at, setAt] = React.useState<Position>({ scenario: 0, step: 0 })
  const [playing, setPlaying] = React.useState(true)
  const isFullscreen = useIsFullscreen()
  // Bumped to restart the current step's animation.
  const [replay, setReplay] = React.useState(0)
  const [elapsed, setElapsed] = React.useState(0)

  const scenario = SCENARIOS[at.scenario]
  const step = scenario.steps[at.step]
  const isLastStep = at.step === scenario.steps.length - 1
  const isVeryLast = isLastStep && at.scenario === SCENARIOS.length - 1

  // The progress timer re-renders every frame; the step's animation must not.
  const visual = React.useMemo(() => {
    const Visual = SCENARIOS[at.scenario].steps[at.step].Visual
    return <Visual key={replay} />
  }, [at, replay])

  const go = React.useCallback((next: Position) => {
    const scenarioIndex = Math.max(
      0,
      Math.min(SCENARIOS.length - 1, next.scenario),
    )
    const steps = SCENARIOS[scenarioIndex].steps.length
    setAt({
      scenario: scenarioIndex,
      step: Math.max(0, Math.min(steps - 1, next.step)),
    })
    setElapsed(0)
    setReplay((count) => count + 1)
  }, [])

  /** One step forward or back, running on into the next walkthrough. */
  const move = React.useCallback(
    (delta: 1 | -1) => {
      const steps = SCENARIOS[at.scenario].steps.length
      const target = at.step + delta
      if (target >= steps && at.scenario < SCENARIOS.length - 1) {
        go({ scenario: at.scenario + 1, step: 0 })
      } else if (target < 0 && at.scenario > 0) {
        go({
          scenario: at.scenario - 1,
          step: SCENARIOS[at.scenario - 1].steps.length - 1,
        })
      } else {
        go({ scenario: at.scenario, step: target })
      }
    },
    [at, go],
  )

  // Autoplay: advance when the step's time is up; stop after the last one.
  React.useEffect(() => {
    if (!playing) return
    let frame = 0
    let last = performance.now()
    const loop = (now: number) => {
      setElapsed((current) => current + (now - last))
      last = now
      frame = requestAnimationFrame(loop)
    }
    frame = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(frame)
  }, [playing])

  React.useEffect(() => {
    if (elapsed < step.duration) return
    if (isVeryLast) {
      setPlaying(false)
      setElapsed(step.duration)
    } else {
      move(1)
    }
  }, [elapsed, step.duration, isVeryLast, move])

  // Presenter keys: arrows or a clicker's PageUp/PageDown step, Shift+arrows
  // switch walkthrough, space plays, digits jump, R replays, F full screen.
  React.useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (
        event.target instanceof HTMLInputElement &&
        event.target.type !== 'range'
      )
        return
      const key = event.key
      // Arrows on a focused walkthrough tab move between the tabs instead.
      if (
        key.startsWith('Arrow') &&
        event.target instanceof Element &&
        event.target.closest('[role="tablist"]')
      )
        return
      if (event.shiftKey && key === 'ArrowRight') {
        go({ scenario: at.scenario + 1, step: 0 })
      } else if (event.shiftKey && key === 'ArrowLeft') {
        go({ scenario: at.scenario - 1, step: 0 })
      } else if (key === 'ArrowRight' || key === 'PageDown') move(1)
      else if (key === 'ArrowLeft' || key === 'PageUp') move(-1)
      else if (key === ' ') {
        event.preventDefault()
        setPlaying((value) => !value)
      } else if (key.toLowerCase() === 'r') go(at)
      else if (key.toLowerCase() === 'f') toggleFullscreen()
      else if (/^[1-9]$/.test(key)) {
        go({ scenario: at.scenario, step: Number(key) - 1 })
      } else return
      event.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [at, go, move])

  const togglePlay = () => {
    if (!playing && isVeryLast && elapsed >= step.duration) {
      go({ scenario: 0, step: 0 })
    }
    setPlaying((value) => !value)
  }

  return (
    <main className="mx-auto grid max-w-[88rem] gap-6 px-4 py-6 sm:px-6 lg:h-[calc(100dvh-3.5rem)] lg:grid-cols-[17rem_minmax(0,1fr)] lg:gap-10 lg:py-8">
      <aside className="flex min-h-0 flex-col gap-5">
        <h1 className="font-heading text-2xl leading-tight font-semibold tracking-tight">
          How SatQuery AI answers a question
        </h1>

        <Tabs
          value={at.scenario}
          onValueChange={(index: number) => go({ scenario: index, step: 0 })}
        >
          <TabsList
            aria-label="Walkthroughs"
            className="grid w-full grid-cols-3 gap-1 group-data-horizontal/tabs:h-auto lg:grid-cols-1"
          >
            {SCENARIOS.map((item, index) => (
              <TabsTrigger
                key={item.id}
                value={index}
                className="h-auto min-w-0 justify-start gap-3 p-1.5 text-start"
              >
                <img
                  src={item.thumbnail}
                  alt=""
                  className="hidden size-10 shrink-0 rounded-md object-cover sm:block"
                />
                <span className="flex min-w-0 flex-col">
                  <span className="truncate">{item.input}</span>
                  <span className="truncate text-xs font-normal text-muted-foreground">
                    {item.model}
                  </span>
                </span>
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>

        <p className="text-sm/6 text-muted-foreground">
          A real, recorded run. {scenario.summary}
        </p>

        <ol className="flex gap-1 overflow-x-auto lg:flex-col lg:overflow-y-auto">
          {scenario.steps.map((item, position) => {
            const state =
              position < at.step
                ? 'done'
                : position === at.step
                  ? 'now'
                  : 'next'
            return (
              <li key={item.title} className="shrink-0">
                <Button
                  variant={state === 'now' ? 'secondary' : 'ghost'}
                  onClick={() => go({ scenario: at.scenario, step: position })}
                  aria-current={state === 'now' ? 'step' : undefined}
                  className="h-auto w-full justify-start gap-3 px-1.5 py-1.5 text-start font-normal whitespace-normal"
                >
                  <Badge
                    variant={state === 'next' ? 'outline' : 'default'}
                    className="size-6 p-0 tabular-nums"
                  >
                    {position + 1}
                  </Badge>
                  <span
                    className={cn(
                      'hidden min-w-0 flex-1 lg:block',
                      state === 'next' && 'text-muted-foreground',
                      state === 'now' && 'font-medium',
                    )}
                  >
                    {item.title}
                  </span>
                  {item.took != null && (
                    <span className="hidden text-xs text-muted-foreground tabular-nums lg:block">
                      {formatSeconds(item.took)}
                    </span>
                  )}
                </Button>
              </li>
            )
          })}
        </ol>

        <div className="mt-auto hidden flex-col gap-3 lg:flex">
          <p className="text-xs/6 text-muted-foreground">
            <Kbd>Space</Kbd> to pause, arrows or a clicker to step,{' '}
            <KbdGroup>
              <Kbd>Shift</Kbd>+ arrows
            </KbdGroup>{' '}
            to switch walkthrough, <Kbd>R</Kbd> to replay, <Kbd>F</Kbd> for full
            screen.
          </p>
          <Button
            variant="link"
            nativeButton={false}
            render={<Link to="/" />}
            className="w-fit px-0"
          >
            Try it with your own images
          </Button>
        </div>
      </aside>

      <section className="flex min-h-0 min-w-0 flex-col gap-5">
        <AnimatePresence mode="wait">
          <motion.header
            key={`title-${at.scenario}-${at.step}`}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
          >
            <p className="text-sm text-muted-foreground tabular-nums">
              {scenario.model}, step {at.step + 1} of {scenario.steps.length}
            </p>
            <h2 className="mt-1 font-heading text-4xl leading-[1.05] font-semibold tracking-[-0.02em] text-balance sm:text-5xl">
              {step.title}
            </h2>
            <p className="mt-3 max-w-3xl text-base/7 text-pretty text-muted-foreground">
              {step.body}
            </p>
          </motion.header>
        </AnimatePresence>

        <div className="graticule relative min-h-[28rem] flex-1 overflow-y-auto rounded-2xl p-4 ring-1 ring-foreground/10 sm:p-6 lg:min-h-0">
          <AnimatePresence mode="wait">
            <motion.div
              key={`${at.scenario}-${at.step}-${replay}`}
              className="h-full"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.25 }}
            >
              {visual}
            </motion.div>
          </AnimatePresence>
        </div>

        <div className="flex items-center gap-2 sm:gap-3">
          <Button
            variant="outline"
            size="icon"
            aria-label="Previous step"
            disabled={at.scenario === 0 && at.step === 0}
            onClick={() => move(-1)}
          >
            <ChevronLeftIcon />
          </Button>
          <Button
            size="icon"
            aria-label={playing ? 'Pause' : 'Play'}
            onClick={togglePlay}
          >
            {playing ? <PauseIcon /> : <PlayIcon />}
          </Button>
          <Button
            variant="outline"
            size="icon"
            aria-label="Next step"
            disabled={isVeryLast}
            onClick={() => move(1)}
          >
            <ChevronRightIcon />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            aria-label="Replay this step"
            onClick={() => go(at)}
          >
            <RotateCcwIcon />
          </Button>

          <div className="flex min-w-0 flex-1 gap-1" aria-hidden>
            {scenario.steps.map((item, position) => (
              <Progress
                key={item.title}
                value={
                  position < at.step
                    ? 100
                    : position > at.step
                      ? 0
                      : Math.min(100, (elapsed / step.duration) * 100)
                }
                // Advanced every frame; easing would make it trail the clock.
                className="flex-1 **:data-[slot=progress-indicator]:transition-none"
              />
            ))}
          </div>

          <Button
            variant="ghost"
            size="icon"
            aria-label={isFullscreen ? 'Exit full screen' : 'Full screen'}
            aria-pressed={isFullscreen}
            className="hidden sm:inline-flex"
            onClick={() => toggleFullscreen()}
          >
            {isFullscreen ? <MinimizeIcon /> : <MaximizeIcon />}
          </Button>
        </div>
      </section>
    </main>
  )
}
