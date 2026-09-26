/**
 * Optical–radar fusion walkthrough: a Sentinel-2 and a Sentinel-1 chip of the
 * same flooded stretch of the Brahmaputra in Assam, from the hand-labelled
 * Sen1Floods11 set (chip india-900498), so the water estimate can be checked.
 */
import { ArrowRightIcon, CpuIcon, InfoIcon } from 'lucide-react'
import { motion, useReducedMotion } from 'motion/react'

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
  useAfter,
} from '#/components/demo/kit'
import type { RecordedRun } from '#/components/demo/kit'
import type { Scenario } from '#/components/demo/scenario'
import recorded from './flood-run.json'

const run = recorded as unknown as RecordedRun & {
  ground_truth: {
    water_share: number
    radar_mask_share: number
    source: string
    bounds: [number, number, number, number]
  }
}
const images = {
  optical: '/demo/flood-optical.jpg',
  radar: '/demo/flood-radar.jpg',
  radarWater: '/demo/flood-radar-water.png',
  labelWater: '/demo/flood-label-water.png',
}
const truth = run.ground_truth

/** "…features: agricultural land 35%, grassland 31%, …" -> [[name, 0.35], …] */
const terrain = Array.from(
  run.evidence[0].description.matchAll(/([a-z ]+?) (\d+)%/g),
  (match) =>
    [match[1].replace(/^.*: /, '').trim(), Number(match[2]) / 100] as const,
)
const similarity = Number(
  run.evidence[1].description.match(/similarity is ([\d.]+)/)?.[1] ?? 0,
)
const opticalOnly =
  Number(run.evidence[2].description.match(/(\d+)% of optical pixels/)?.[1]) /
  100

/** Fusion checkpoint facts (checkpoints/fusion_best.pt) and our own check. */
const model = {
  backbone: 'ResNet-50',
  /** Validation: a radar patch's embedding finds its own optical twin. */
  matching: 0.847,
  terrainClasses: 4,
  /** Mean gap between the radar water share and the hand label, 8 chips. */
  waterError: 0.057,
  chips: 8,
}

const pct = (value: number) => `${Math.round(value * 100)}%`

// --- Two views, one model -----------------------------------------------------------

function EncodeStep() {
  const reduceMotion = useReducedMotion()
  return (
    <div className="flex h-full flex-col justify-center gap-8">
      <div className="grid items-center gap-4 lg:grid-cols-[10rem_auto_1fr_auto_12rem]">
        <div className="flex gap-3 lg:flex-col">
          {[
            { src: images.optical, label: 'Optical' },
            { src: images.radar, label: 'Radar' },
          ].map((view, index) => (
            <Appear
              key={view.label}
              delay={0.2 + index * 0.2}
              className="flex-1"
            >
              <img
                src={view.src}
                alt={`${view.label} image`}
                className="aspect-square w-full rounded-lg object-cover ring-1 ring-foreground/10"
              />
              <p className="mt-1 text-xs text-muted-foreground">{view.label}</p>
            </Appear>
          ))}
        </div>
        <Arrow delay={0.8} />
        <Appear
          delay={1}
          className="rounded-xl bg-card p-5 ring-1 ring-foreground/10"
        >
          <p className="flex items-center gap-2 font-heading text-lg font-semibold">
            <CpuIcon className="size-5" />
            Dual encoder, fine-tuned
          </p>
          <p className="mt-2 text-sm/6 text-muted-foreground">
            Two {model.backbone} branches, one per sensor, trained so that an
            optical patch and a radar patch of the same ground land close
            together. A small head on top names the land type.
          </p>
        </Appear>
        <Arrow delay={1.6} />
        <Appear
          delay={1.8}
          className="rounded-xl bg-card p-5 ring-1 ring-foreground/10"
        >
          <p className="text-sm text-muted-foreground">
            How well the two views agree
          </p>
          <motion.p
            className="font-heading text-5xl font-semibold"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: reduceMotion ? 0 : 2 }}
          >
            {similarity.toFixed(2)}
          </motion.p>
          <p className="text-xs text-muted-foreground">
            Cosine similarity of the two embeddings; 1 means they agree
            completely
          </p>
        </Appear>
      </div>

      <Appear
        delay={2.6}
        className="flex items-center gap-4 rounded-xl bg-card p-5 ring-1 ring-foreground/10"
      >
        <p className="font-heading text-4xl font-semibold">
          {(model.matching * 100).toFixed(1)}%
        </p>
        <p className="text-sm/6 text-muted-foreground">
          of held-out radar patches matched to their own optical twin during
          validation: the encoder has learned what the same ground looks like to
          both sensors.
        </p>
      </Appear>
    </div>
  )
}

function Arrow({ delay }: { delay: number }) {
  return (
    <motion.span
      initial={{ opacity: 0, x: -6 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay, duration: 0.4 }}
      className="hidden text-muted-foreground lg:block"
    >
      <ArrowRightIcon className="size-5" />
    </motion.span>
  )
}

// --- Classify the land -----------------------------------------------------------

function TerrainStep() {
  return (
    <div className="grid h-full items-center gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <Frame src={images.optical} alt="The optical image" size="md" />
      <div className="flex flex-col gap-4">
        {terrain.map(([name, value], index) => (
          <div key={name}>
            <div className="flex items-baseline justify-between text-base">
              <span className={index === 0 ? 'font-semibold' : ''}>
                {name[0].toUpperCase() + name.slice(1)}
              </span>
              <span className="tabular-nums">{pct(value)}</span>
            </div>
            <div className="mt-1.5 h-2.5 overflow-hidden rounded-full bg-muted">
              <motion.div
                className={`h-full rounded-full ${index === 0 ? 'bg-foreground' : 'bg-foreground/35'}`}
                initial={{ width: 0 }}
                animate={{ width: `${value * 100}%` }}
                transition={{
                  delay: 0.4 + index * 0.25,
                  duration: 0.8,
                  ease: EASE,
                }}
              />
            </div>
          </div>
        ))}
        <Appear
          delay={1.8}
          className="mt-2 flex items-start gap-2 text-sm/6 text-muted-foreground"
        >
          <InfoIcon className="mt-1 size-4 shrink-0" />
          This checkpoint knows {model.terrainClasses} land types. Water is
          measured separately, in the next step.
        </Appear>
      </div>
    </div>
  )
}

// --- Measure the water -------------------------------------------------------------

function WaterStep() {
  const reduceMotion = useReducedMotion()
  const labelShown = useAfter(2600)
  return (
    <div className="grid h-full items-center gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_15rem]">
      <div>
        <Frame
          src={images.radar}
          alt="Radar image with the water it detected"
          size="md"
        >
          <motion.div
            className="absolute inset-0"
            initial={{
              clipPath: reduceMotion ? 'inset(0 0 0 0)' : 'inset(0 0 100% 0)',
            }}
            animate={{ clipPath: 'inset(0 0 0 0)' }}
            transition={{ delay: 0.5, duration: 1.8, ease: 'easeInOut' }}
          >
            <MaskLayer src={images.radarWater} opacity={0.75} />
          </motion.div>
        </Frame>
        <p className="mt-3 text-sm font-medium">From the radar</p>
      </div>
      <div className={labelShown ? '' : 'invisible'}>
        {labelShown && (
          <Appear>
            <Frame src={images.optical} alt="Hand-labelled water" size="md">
              <MaskLayer
                src={images.labelWater}
                opacity={0.75}
                className="bg-sky-500"
              />
            </Frame>
            <p className="mt-3 text-sm font-medium">
              Drawn by hand (ground truth)
            </p>
          </Appear>
        )}
      </div>
      <div className="flex flex-col gap-4">
        <Appear delay={2}>
          <p className="font-heading text-5xl font-semibold text-fuchsia-500 tabular-nums">
            {pct(truth.radar_mask_share)}
          </p>
          <p className="text-sm text-muted-foreground">
            under water, from the radar
          </p>
        </Appear>
        {labelShown && (
          <>
            <Appear>
              <p className="font-heading text-5xl font-semibold text-sky-500 tabular-nums">
                {pct(truth.water_share)}
              </p>
              <p className="text-sm text-muted-foreground">
                in the hand-drawn label (Sen1Floods11)
              </p>
            </Appear>
            <Appear
              delay={0.4}
              className="flex items-start gap-2 border-t pt-4 text-sm/6"
            >
              <CheckBadge className="mt-0.5" />
              <span>
                Across all {model.chips} hand-labelled chips we checked, the
                radar is {(model.waterError * 100).toFixed(1)} points off on
                average. From the optical image alone the estimate here is{' '}
                {pct(opticalOnly)}: without an infrared band it over-counts, so
                the radar figure is the one to trust.
              </span>
            </Appear>
          </>
        )}
      </div>
    </div>
  )
}

// --- Answer ------------------------------------------------------------------------

function AnswerStep() {
  return (
    <div className="grid h-full items-start gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
      <Appear className="grid grid-cols-2 gap-3">
        <img
          src={images.optical}
          alt="Optical image"
          className="aspect-square w-full rounded-lg object-cover ring-1 ring-foreground/10"
        />
        <div className="relative aspect-square overflow-hidden rounded-lg ring-1 ring-foreground/10">
          <img
            src={images.radar}
            alt="Radar image with the detected water"
            className="size-full object-cover"
          />
          <MaskLayer src={images.radarWater} opacity={0.6} />
        </div>
      </Appear>
      <div className="flex flex-col gap-5">
        <Appear delay={0.3}>
          <p className="text-base/7 whitespace-pre-line">{run.answer}</p>
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

const [south, west] = [truth.bounds[1], truth.bounds[0]]

export const fusionScenario: Scenario = {
  id: 'fusion',
  input: 'Optical + radar',
  model: 'Optical–radar fusion',
  summary: `A flooded stretch of the Brahmaputra in Assam (${south.toFixed(1)}° N, ${west.toFixed(1)}° E), seen by an optical and a radar satellite.`,
  thumbnail: images.radar,
  steps: [
    {
      title: 'Ask about both images',
      body: 'One question about a pair: a normal photo and a radar image of the same place.',
      duration: 7500,
      Visual: () => <AskStep run={run} />,
    },
    {
      title: 'Check the images',
      body: 'The controller tells the two sensors apart: Sentinel-2 photographs the ground, Sentinel-1 radar sees through cloud and shows water as dark.',
      duration: 7000,
      took: traceStep(run, 'validate')?.duration_ms,
      Visual: () => (
        <InputsStep
          shots={[
            {
              src: images.optical,
              info: run.inputs[0],
              title: 'Optical',
              caption: 'Sentinel-2',
              modality: 'Optical',
            },
            {
              src: images.radar,
              info: run.inputs[1],
              title: 'Radar',
              caption: 'Sentinel-1',
              modality: 'Radar (SAR)',
            },
          ]}
          verdict={
            <>
              An optical and a radar image of the same place:{' '}
              <span className="font-semibold">a cross-modal pair</span>.
            </>
          }
        />
      ),
    },
    {
      title: 'Pick the task',
      body: 'An optical image and a radar image of one place go to fusion, by the same fixed rules.',
      duration: 6500,
      took: traceStep(run, 'classify')?.duration_ms,
      Visual: () => <RouteStep run={run} />,
    },
    {
      title: 'Two views, one model',
      body: 'Both images go through a model fine-tuned on paired optical and radar patches, so it can compare what each sensor sees.',
      duration: 8500,
      took: traceStep(run, 'execute')?.duration_ms,
      Visual: EncodeStep,
    },
    {
      title: 'Classify the land',
      body: 'The fused features give a probability for each land type.',
      duration: 6500,
      Visual: TerrainStep,
    },
    {
      title: 'Measure the water',
      body: 'Radar marks smooth water as dark, through cloud and haze. Here the result is checked against a map of the water drawn by hand.',
      duration: 10000,
      Visual: WaterStep,
    },
    {
      title: 'The answer',
      body: `The land type, how much is under water, and a record of every step, in ${formatSeconds(run.execution_time_ms)}.`,
      duration: 12000,
      took: traceStep(run, 'aggregate')?.duration_ms,
      Visual: AnswerStep,
    },
  ],
}
