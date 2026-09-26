/**
 * Grounding walkthrough: one photo with three storage tanks, and a question
 * that singles one of them out. The image and its hand-labelled answer box are
 * from the DIOR-RSVG test split (image 11074); the model was fine-tuned on
 * VRSBench, a different benchmark.
 */
import { ArrowRightIcon, CpuIcon } from 'lucide-react'
import { motion } from 'motion/react'

import {
  Appear,
  AskStep,
  BoxOutline,
  CheckBadge,
  ConfidenceRow,
  EASE,
  Frame,
  InputsStep,
  RouteStep,
  StepTimeline,
  formatSeconds,
  traceStep,
  useAfter,
} from '#/components/demo/kit'
import type { RecordedRun } from '#/components/demo/kit'
import type { Scenario } from '#/components/demo/scenario'
import type { BoundingBox } from '#/lib/api'
import recorded from './tanks-run.json'

type Box = BoundingBox

const run = recorded as unknown as RecordedRun & {
  ground_truth: { text: string; box: Box; source: string }
}
const image = '/demo/tanks.jpg'
const found = run.evidence.find((item) => item.type === 'bbox')!
const foundBox = found.data as Box
const truth = run.ground_truth

/** From results/grounding_reranked_ensemble_full.json (VRSBench validation). */
const benchmark = {
  queries: 16146,
  accuracy: 0.675,
  storageTanks: 0.851,
  published: [
    { name: 'GeoChat', accuracy: 0.574 },
    { name: 'GeoGround', accuracy: 0.66 },
  ],
}

function iou(a: Box, b: Box) {
  const x = Math.max(0, Math.min(a.x_max, b.x_max) - Math.max(a.x_min, b.x_min))
  const y = Math.max(0, Math.min(a.y_max, b.y_max) - Math.max(a.y_min, b.y_min))
  const inter = x * y
  const area = (box: Box) => (box.x_max - box.x_min) * (box.y_max - box.y_min)
  return inter / (area(a) + area(b) - inter)
}
const overlap = iou(foundBox, truth.box)

const percent = (value: number) => `${(value * 100).toFixed(1)}%`

// --- Model -------------------------------------------------------------------------

const STAGES = [
  {
    title: 'Detector',
    body: 'Fine-tuned GroundingDINO proposes the 10 likeliest boxes for the sentence.',
  },
  {
    title: '5 re-rankers',
    body: 'Each reads the whole sentence, including “upper left”, and re-scores the 10 boxes; their votes are averaged.',
  },
  {
    title: 'Best box',
    body: 'The top-scoring box is the answer, with its score as the confidence.',
  },
]

function ModelStep() {
  return (
    <div className="flex h-full flex-col justify-center gap-8">
      <div className="grid items-stretch gap-3 lg:grid-cols-[1fr_auto_1fr_auto_1fr]">
        {STAGES.map((stage, index) => (
          <Stage key={stage.title} index={index} {...stage} />
        ))}
      </div>

      <Appear
        delay={2.2}
        className="grid gap-6 rounded-xl bg-card p-5 ring-1 ring-foreground/10 sm:grid-cols-[auto_1fr]"
      >
        <div>
          <p className="font-heading text-5xl font-semibold">
            {percent(benchmark.accuracy)}
          </p>
          <p className="mt-1 max-w-56 text-sm text-muted-foreground">
            of {benchmark.queries.toLocaleString('en-IN')} VRSBench validation
            questions answered with the right box
          </p>
        </div>
        <div className="flex flex-col justify-center gap-2 text-sm">
          <Bar label="SatQuery AI" value={benchmark.accuracy} strong />
          {benchmark.published.map((entry) => (
            <Bar key={entry.name} label={entry.name} value={entry.accuracy} />
          ))}
          <p className="mt-1 text-xs text-muted-foreground">
            Published results on the same benchmark. For storage tanks alone,
            SatQuery AI gets {percent(benchmark.storageTanks)}.
          </p>
        </div>
      </Appear>
    </div>
  )
}

function Stage({
  index,
  title,
  body,
}: {
  index: number
  title: string
  body: string
}) {
  return (
    <>
      {index > 0 && (
        <motion.span
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 + index * 0.6 }}
          className="hidden items-center text-muted-foreground lg:flex"
        >
          <ArrowRightIcon className="size-5" />
        </motion.span>
      )}
      <Appear
        delay={0.3 + index * 0.6}
        className="rounded-xl bg-card p-5 ring-1 ring-foreground/10"
      >
        <p className="flex items-center gap-2 font-heading text-lg font-semibold">
          {index === 0 && <CpuIcon className="size-5" />}
          {title}
        </p>
        <p className="mt-2 text-sm/6 text-muted-foreground">{body}</p>
      </Appear>
    </>
  )
}

function Bar({
  label,
  value,
  strong = false,
}: {
  label: string
  value: number
  strong?: boolean
}) {
  return (
    <div className="grid grid-cols-[6.5rem_minmax(0,1fr)_3.5rem] items-center gap-3">
      <span className={strong ? 'font-medium' : 'text-muted-foreground'}>
        {label}
      </span>
      <span className="h-2 overflow-hidden rounded-full bg-muted">
        <motion.span
          className={`block h-full rounded-full ${strong ? 'bg-fuchsia-500' : 'bg-foreground/40'}`}
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ delay: 2.5, duration: 0.8, ease: EASE }}
        />
      </span>
      <span className="text-end tabular-nums">{percent(value)}</span>
    </div>
  )
}

// --- Find it -------------------------------------------------------------------------

function FindStep() {
  const boxShown = useAfter(1800)
  const truthShown = useAfter(3400)

  return (
    <div className="grid h-full items-center gap-8 lg:grid-cols-[minmax(0,1fr)_17rem]">
      <Frame src={image} alt="Three storage tanks, seen from above">
        {!boxShown && (
          <motion.div
            aria-hidden
            className="absolute inset-x-0 h-24 bg-gradient-to-b from-transparent to-fuchsia-500/25"
            initial={{ top: '-25%' }}
            animate={{ top: '100%' }}
            transition={{ duration: 1.6, ease: 'easeInOut' }}
          >
            <div className="absolute inset-x-0 bottom-0 h-px bg-fuchsia-500 shadow-[0_0_12px_2px] shadow-fuchsia-500/60" />
          </motion.div>
        )}
        {boxShown && (
          <motion.div
            initial={{ opacity: 0, scale: 1.2 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, ease: EASE }}
            className="absolute inset-0"
          >
            <BoxOutline
              box={foundBox}
              className="rounded-[2px] border-[3px] border-fuchsia-500 shadow-[0_0_0_1px_rgb(0_0_0/0.5)]"
            >
              <span className="absolute -top-px left-0 -translate-y-full rounded-t-sm bg-fuchsia-500 px-1.5 py-0.5 text-xs font-semibold whitespace-nowrap text-white tabular-nums">
                1 · {Math.round((found.confidence ?? 0) * 100)}%
              </span>
            </BoxOutline>
          </motion.div>
        )}
        {truthShown && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="absolute inset-0"
          >
            <BoxOutline
              box={truth.box}
              className="border-2 border-dashed border-white"
            />
          </motion.div>
        )}
      </Frame>

      <div className="flex flex-col gap-5">
        <p className="text-lg/7">“{run.query}”</p>
        {boxShown && (
          <Appear className="flex items-start gap-3">
            <span className="mt-1 size-3 shrink-0 rounded-sm bg-fuchsia-500" />
            <p className="text-sm/6">
              The model’s answer, {Math.round((found.confidence ?? 0) * 100)}%
              sure. It picked the upper-left tank, not the other two.
            </p>
          </Appear>
        )}
        {truthShown && (
          <Appear className="flex flex-col gap-4">
            <div className="flex items-start gap-3">
              <span className="mt-1 size-3 shrink-0 rounded-sm border-2 border-dashed border-foreground" />
              <p className="text-sm/6">
                The hand-labelled answer from the DIOR-RSVG test set, a
                benchmark the model was not trained on.
              </p>
            </div>
            <div className="flex items-center gap-3 border-t pt-4">
              <CheckBadge />
              <p>
                <span className="font-heading text-3xl font-semibold tabular-nums">
                  {Math.round(overlap * 100)}%
                </span>{' '}
                <span className="text-sm text-muted-foreground">
                  overlap (IoU); 50% counts as correct
                </span>
              </p>
            </div>
            <p className="text-sm text-muted-foreground tabular-nums">
              Found in {formatSeconds(traceStep(run, 'execute')?.duration_ms)}
            </p>
          </Appear>
        )}
      </div>
    </div>
  )
}

// --- Answer ------------------------------------------------------------------------

function AnswerStep() {
  return (
    <div className="grid h-full items-start gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <Appear>
        <Frame src={image} alt="The storage tank the model found">
          <BoxOutline
            box={foundBox}
            className="rounded-[2px] border-[3px] border-fuchsia-500 shadow-[0_0_0_1px_rgb(0_0_0/0.5)]"
          >
            <span className="absolute -top-px left-0 flex size-5 -translate-y-full items-center justify-center rounded-t-sm bg-fuchsia-500 text-xs font-semibold text-white">
              1
            </span>
          </BoxOutline>
        </Frame>
      </Appear>
      <div className="flex flex-col gap-5">
        <Appear delay={0.3}>
          <p className="font-heading text-2xl/9 font-medium">{run.answer}</p>
          <p className="mt-2 text-sm text-muted-foreground">
            {run.trace.warnings[0]}
          </p>
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

export const groundingScenario: Scenario = {
  id: 'grounding',
  input: 'One image',
  model: 'Grounding',
  summary:
    'Three storage tanks in one photo, and a question that picks out one of them.',
  thumbnail: image,
  steps: [
    {
      title: 'Ask where something is',
      body: 'The question names one object among several look-alikes, by its position.',
      duration: 6500,
      Visual: () => <AskStep run={run} />,
    },
    {
      title: 'Check the image',
      body: 'One photo, with no map coordinates in the file, so locations are given within the image itself.',
      duration: 6000,
      took: traceStep(run, 'validate')?.duration_ms,
      Visual: () => (
        <InputsStep
          shots={[
            {
              src: image,
              info: run.inputs[0],
              title: 'Storage tanks',
              caption: 'DIOR-RSVG test image',
              modality: 'Optical',
            },
          ]}
          verdict={
            <>
              One image, and the question asks{' '}
              <span className="font-semibold">where</span> something is.
            </>
          }
        />
      ),
    },
    {
      title: 'Pick the task',
      body: 'For a single image, a question that asks where something is, or to find, show or mark it, goes to grounding. Plain word rules, no model.',
      duration: 7000,
      took: traceStep(run, 'classify')?.duration_ms,
      Visual: () => <RouteStep run={run} />,
    },
    {
      title: 'Two stages, one answer',
      body: 'A detector fine-tuned on remote-sensing images proposes candidates; a panel of re-rankers reads the whole sentence and picks one.',
      duration: 8000,
      took: traceStep(run, 'select')?.duration_ms,
      Visual: ModelStep,
    },
    {
      title: 'Find it',
      body: 'The model returns one box. Here it is checked against the benchmark’s own hand-labelled answer.',
      duration: 8000,
      took: traceStep(run, 'execute')?.duration_ms,
      Visual: FindStep,
    },
    {
      title: 'The answer',
      body: 'The box, its confidence, and a record of every step.',
      duration: 10000,
      took: traceStep(run, 'aggregate')?.duration_ms,
      Visual: AnswerStep,
    },
  ],
}
