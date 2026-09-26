/**
 * Change-VQA walkthrough: the Assam Brahmaputra pair (dry season, 28 Jan 2024
 * vs monsoon, 26 Jul 2024), asked about in Hindi.
 *
 * Re-record by POSTing the pair to /api/analyse with language=hi and saving
 * the response as assam-run.json, with the preview/mask data URIs moved to
 * public/demo/.
 */
import * as React from 'react'
import { ArrowRightIcon, BanIcon, CpuIcon } from 'lucide-react'
import { motion, useReducedMotion } from 'motion/react'
import { cn } from 'cn'

import { ChangeSwipe } from '#/components/change-swipe'
import {
  Appear,
  AskStep,
  CheckBadge,
  ConfidenceRow,
  EASE,
  Frame,
  InputsStep,
  MaskLayer,
  RouteStep,
  StepTimeline,
  formatSeconds,
  traceStep,
  useTicker,
} from '#/components/demo/kit'
import type { RecordedRun } from '#/components/demo/kit'
import type { Scenario } from '#/components/demo/scenario'
import type { BoundingBox } from '#/lib/api'
import recorded from './assam-run.json'

const run = recorded as unknown as RecordedRun

const images = {
  before: '/demo/assam-before.jpg',
  after: '/demo/assam-after.jpg',
  mask: '/demo/assam-mask.png',
}
const dates = { before: '28 Jan 2024', after: '26 Jul 2024' }

/** Facts about the change model, from its checkpoint. */
const model = {
  name: 'Change-VQA',
  backbone: 'Siamese ResNet-34',
  trainedGsd: [0.5, 20] as const,
  tolerance: 2,
  /** 1536 px at 10 m, downscaled to 1280 px before analysis. */
  analysisGsd: (10 * 1536) / 1280,
  tileSize: 256,
  tileOverlap: 32,
  answers: 60,
  sources: ['Sentinel-2 + Dynamic World (10 m)', 'LEVIR-CD (0.5 m)'],
}

/** 6 x 6: 256 px tiles every 224 px across 1280 px, the last one flush. */
const TILES_PER_SIDE = 6

const mask = run.evidence[0]
const regions = run.evidence
  .map((item, index) => ({ ...item, number: index + 1 }))
  .filter((item) => item.type === 'bbox')
  .map((item) => ({ ...item, box: item.data as BoundingBox }))

// --- Model ------------------------------------------------------------------------

const GSD_MIN = 0.1
const GSD_MAX = 100
/** Position of a resolution on a log scale from 0.1 m to 100 m, 0-100%. */
function gsdPosition(metres: number) {
  return (
    ((Math.log10(metres) - Math.log10(GSD_MIN)) /
      (Math.log10(GSD_MAX) - Math.log10(GSD_MIN))) *
    100
  )
}

function ModelStep() {
  const [low, high] = model.trainedGsd
  const tolerated = [low / model.tolerance, high * model.tolerance]
  const reduceMotion = useReducedMotion()
  const [landed, setLanded] = React.useState(Boolean(reduceMotion))

  return (
    <div className="flex h-full flex-col justify-center gap-8">
      <div className="grid gap-4 sm:grid-cols-2">
        <motion.div
          animate={{ opacity: landed ? 1 : 0.8 }}
          className={cn(
            'rounded-xl p-5 ring-1 transition-shadow duration-500',
            landed
              ? 'bg-card ring-2 ring-foreground'
              : 'bg-card ring-foreground/10',
          )}
        >
          <div className="flex items-center gap-2">
            <CpuIcon className="size-5" />
            <p className="font-heading text-lg font-semibold">
              Fine-tuned {model.name}
            </p>
            {landed && <CheckBadge className="ms-auto" />}
          </div>
          <p className="mt-2 text-sm/6 text-muted-foreground">
            {model.backbone}, trained on {model.sources.join(' and ')} pairs.
            Knows {model.answers} kinds of change.
          </p>
        </motion.div>
        <motion.div
          animate={{ opacity: landed ? 0.45 : 0.8 }}
          className="rounded-xl bg-card p-5 ring-1 ring-foreground/10"
        >
          <div className="flex items-center gap-2">
            <BanIcon className="size-5" />
            <p className="font-heading text-lg font-semibold">
              Outside the range
            </p>
          </div>
          <p className="mt-2 text-sm/6 text-muted-foreground">
            Imagery far outside the trained range is not given to the fine-tuned
            model, and the result carries a warning saying why.
          </p>
        </motion.div>
      </div>

      <div className="rounded-xl bg-card p-5 ring-1 ring-foreground/10">
        <p className="text-sm text-muted-foreground">
          Is this image close to what the model was trained on?
        </p>
        <div className="relative mt-10 mb-8 h-3 rounded-full bg-muted">
          <div
            className="absolute inset-y-0 rounded-full bg-foreground/15"
            style={{
              left: `${gsdPosition(tolerated[0])}%`,
              right: `${100 - gsdPosition(tolerated[1])}%`,
            }}
          />
          <div
            className="absolute inset-y-0 rounded-full bg-foreground/45"
            style={{
              left: `${gsdPosition(low)}%`,
              right: `${100 - gsdPosition(high)}%`,
            }}
          />
          <motion.div
            className="absolute top-1/2 flex -translate-x-1/2 -translate-y-1/2 flex-col items-center"
            initial={{
              left: reduceMotion ? `${gsdPosition(model.analysisGsd)}%` : '98%',
            }}
            animate={{ left: `${gsdPosition(model.analysisGsd)}%` }}
            transition={{ delay: 0.6, duration: 1.6, ease: EASE }}
            onAnimationComplete={() => setLanded(true)}
          >
            <span className="absolute bottom-full mb-2 rounded-md bg-fuchsia-500 px-2 py-0.5 text-xs font-semibold whitespace-nowrap text-white tabular-nums">
              This image: {model.analysisGsd} m
            </span>
            <span className="size-5 rounded-full bg-fuchsia-500 ring-4 ring-card" />
          </motion.div>
          {[0.1, 1, 10, 100].map((tick) => (
            <span
              key={tick}
              className="absolute top-full mt-2 -translate-x-1/2 text-xs whitespace-nowrap text-muted-foreground tabular-nums"
              style={{ left: `${gsdPosition(tick)}%` }}
            >
              {tick} m
            </span>
          ))}
        </div>
        <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-muted-foreground">
          <span className="flex items-center gap-1.5">
            <span className="size-2.5 rounded-full bg-foreground/45" />
            Trained on {low}–{high} m per pixel
          </span>
          <span className="flex items-center gap-1.5">
            <span className="size-2.5 rounded-full bg-foreground/15" />
            Accepted, up to {model.tolerance}× either side
          </span>
        </div>
      </div>
    </div>
  )
}

// --- Find the change --------------------------------------------------------------

function ChangeStep() {
  const total = TILES_PER_SIDE * TILES_PER_SIDE
  const scanned = useTicker(total, 85, 500)
  const done = scanned === total - 1
  const changed = mask.description.match(/^([\d.]+%)/)?.[1]
  const area = mask.description.match(/about ([\d.]+ km²)/)?.[1]
  const url = `url("${images.mask}")`

  return (
    <div className="grid h-full items-center gap-8 lg:grid-cols-[minmax(0,1fr)_16rem]">
      <Frame src={images.after} alt="Monsoon image with the change mask">
        {Array.from({ length: total }, (_, index) => {
          const row = Math.floor(index / TILES_PER_SIDE)
          const col = index % TILES_PER_SIDE
          const size = 100 / TILES_PER_SIDE
          return (
            <div
              key={index}
              className="absolute overflow-hidden"
              style={{
                left: `${col * size}%`,
                top: `${row * size}%`,
                width: `${size}%`,
                height: `${size}%`,
              }}
            >
              {index <= scanned && (
                <motion.div
                  aria-hidden
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 0.6 }}
                  transition={{ duration: 0.4 }}
                  className="absolute bg-fuchsia-500"
                  style={{
                    width: `${TILES_PER_SIDE * 100}%`,
                    height: `${TILES_PER_SIDE * 100}%`,
                    left: `${-col * 100}%`,
                    top: `${-row * 100}%`,
                    maskImage: url,
                    WebkitMaskImage: url,
                    maskSize: '100% 100%',
                    WebkitMaskSize: '100% 100%',
                  }}
                />
              )}
              {index === scanned && !done && (
                <div className="absolute inset-0 bg-white/15 ring-2 ring-white ring-inset" />
              )}
            </div>
          )
        })}
      </Frame>

      <div className="flex flex-col gap-6">
        <div className="flex items-center gap-2">
          <img
            src={images.before}
            alt=""
            className="size-14 rounded-md object-cover ring-1 ring-foreground/10"
          />
          <span className="text-muted-foreground">+</span>
          <img
            src={images.after}
            alt=""
            className="size-14 rounded-md object-cover ring-1 ring-foreground/10"
          />
          <ArrowRightIcon className="size-4 text-muted-foreground" />
          <span className="text-sm font-medium">{model.backbone}</span>
        </div>
        <div>
          <p className="font-heading text-5xl font-semibold tabular-nums">
            {Math.max(0, scanned + 1)}
            <span className="text-2xl text-muted-foreground"> / {total}</span>
          </p>
          <p className="mt-1 text-sm text-muted-foreground">
            tiles of {model.tileSize} px, overlapping by {model.tileOverlap} px
          </p>
        </div>
        {done && (
          <Appear className="flex flex-col gap-4 border-t pt-5">
            <div>
              <p className="font-heading text-4xl font-semibold text-fuchsia-500">
                {changed}
              </p>
              <p className="text-sm text-muted-foreground">
                of the ground changed, about {area}
              </p>
            </div>
            <p className="text-sm text-muted-foreground tabular-nums">
              {formatSeconds(traceStep(run, 'execute')?.duration_ms)} on the GPU
            </p>
          </Appear>
        )}
      </div>
    </div>
  )
}

// --- Measure each change -------------------------------------------------------------

/** "Bare ground to water: 33.5% of the scene (about 79 km²), in the south sector." */
function regionArea(description: string) {
  return description.match(/about ([\d.]+ km²)/)?.[1] ?? ''
}

function RegionsStep() {
  const total = mask.description.match(/in (\d+) regions/)?.[1]
  return (
    <div className="grid h-full items-center gap-8 lg:grid-cols-[minmax(0,1fr)_18rem]">
      <Frame src={images.after} alt="Monsoon image with the changed regions">
        <MaskLayer src={images.mask} opacity={0.3} />
        {regions.map((region, index) => (
          <motion.div
            key={region.number}
            initial={{ opacity: 0, scale: 1.15 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{
              delay: 0.5 + index * 0.45,
              duration: 0.45,
              ease: EASE,
            }}
            className="absolute rounded-[2px] border-2 border-white shadow-[0_0_0_1px_rgb(0_0_0/0.6)]"
            style={{
              left: `${region.box.x_min * 100}%`,
              top: `${region.box.y_min * 100}%`,
              width: `${(region.box.x_max - region.box.x_min) * 100}%`,
              height: `${(region.box.y_max - region.box.y_min) * 100}%`,
            }}
          >
            <span className="absolute top-0 left-0 flex size-5 items-center justify-center bg-white text-[11px] font-semibold text-black tabular-nums">
              {region.number}
            </span>
          </motion.div>
        ))}
      </Frame>

      <div className="flex flex-col gap-4">
        <p className="text-base/7">
          <span className="font-semibold tabular-nums">{total}</span> separate
          changed patches. The {regions.length} largest are numbered, and the
          model names the change in each.
        </p>
        <ol className="flex flex-col gap-1.5">
          {regions.map((region, index) => (
            <Appear key={region.number} delay={0.5 + index * 0.45} y={4}>
              <li className="flex items-center gap-2.5 text-sm">
                <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-foreground text-[11px] font-semibold text-background tabular-nums">
                  {region.number}
                </span>
                <span className="min-w-0 flex-1 truncate">{region.label}</span>
                <span className="text-muted-foreground tabular-nums">
                  {regionArea(region.description)}
                </span>
              </li>
            </Appear>
          ))}
        </ol>
      </div>
    </div>
  )
}

// --- Answer ------------------------------------------------------------------------

function AnswerStep() {
  return (
    <div className="grid h-full items-start gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <Appear className="mx-auto w-full lg:max-w-[calc(100dvh-25rem)]">
        <ChangeSwipe
          before={{
            src: images.before,
            alt: `Dry season, ${dates.before}`,
            label: 'Before',
          }}
          after={{
            src: images.after,
            alt: `Monsoon, ${dates.after}`,
            label: 'After',
          }}
          dividerLabel="Drag to compare before and after"
          evidence={run.evidence}
          numberOf={(item) => run.evidence.indexOf(item) + 1}
          maskOpacity={0.4}
          activeIndex={null}
        />
        <p className="mt-2 text-xs text-muted-foreground">
          Drag across the image to compare the two dates.
        </p>
      </Appear>

      <div className="flex flex-col gap-5">
        <Appear delay={0.3}>
          <p lang="hi" className="text-lg/8 font-medium">
            {run.translation?.answer}
          </p>
          <p className="mt-2 text-xs/5 text-muted-foreground">{run.answer}</p>
        </Appear>
        <Appear delay={0.6}>
          <ConfidenceRow run={run} />
        </Appear>
        <Appear delay={0.9}>
          <StepTimeline run={run} />
        </Appear>
      </div>
    </div>
  )
}

// --- Scenario ------------------------------------------------------------------------

export const changeScenario: Scenario = {
  id: 'change',
  input: 'Before and after',
  model: 'Change-VQA',
  summary:
    'Flooding on the Brahmaputra in Assam, dry season against monsoon, asked about in Hindi.',
  thumbnail: images.after,
  steps: [
    {
      title: 'Ask in your own language',
      body: 'The question is typed in Hindi. IndicTrans2 translates it to English for the models, and the answer is translated back at the end.',
      duration: 9000,
      took: traceStep(run, 'translate', 'query-translator')?.duration_ms,
      Visual: () => <AskStep run={run} />,
    },
    {
      title: 'Check the images',
      body: 'The controller reads each file: what kind of image it is, its size, and how much ground each pixel covers. Two optical images of one place become a before/after pair.',
      duration: 7000,
      took: traceStep(run, 'validate')?.duration_ms,
      Visual: () => (
        <InputsStep
          shots={[
            {
              src: images.before,
              info: run.inputs[0],
              title: 'Dry season',
              caption: dates.before,
              modality: 'Optical',
            },
            {
              src: images.after,
              info: run.inputs[1],
              title: 'Monsoon',
              caption: dates.after,
              modality: 'Optical',
            },
          ]}
          verdict={
            <>
              Two optical images of the same place, on different dates:{' '}
              <span className="font-semibold">a before/after pair</span>.
            </>
          }
        />
      ),
    },
    {
      title: 'Pick the task',
      body: 'The kind of input decides the task, by fixed rules in the controller. Two dates of one place means a question about change.',
      duration: 6500,
      took: traceStep(run, 'classify')?.duration_ms,
      Visual: () => <RouteStep run={run} />,
    },
    {
      title: 'Choose the right model',
      body: 'Each task has a model fine-tuned on satellite imagery. It is only used on imagery close to the resolution it was trained on.',
      duration: 7500,
      took: traceStep(run, 'select')?.duration_ms,
      Visual: ModelStep,
    },
    {
      title: 'Find what changed',
      body: 'The fine-tuned model compares the two dates tile by tile and marks every pixel that changed.',
      duration: 7000,
      took: traceStep(run, 'execute')?.duration_ms,
      Visual: ChangeStep,
    },
    {
      title: 'Measure each change',
      body: 'Changed pixels are grouped into regions, and the model names the change in each. The known pixel size turns shares of the image into square kilometres.',
      duration: 7500,
      Visual: RegionsStep,
    },
    {
      title: 'The answer',
      body: 'The fine-tuned model’s answer, translated back to Hindi, with the evidence marked on the image, a confidence score, and a record of every step.',
      duration: 14000,
      took: traceStep(run, 'translate', 'answer-translator')?.duration_ms,
      Visual: AnswerStep,
    },
  ],
}
