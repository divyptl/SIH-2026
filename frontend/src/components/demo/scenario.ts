import type * as React from 'react'

export interface DemoStep {
  title: string
  body: string
  /** How long autoplay stays on the step, in ms. */
  duration: number
  /** Time the step took in the recorded run, in ms. */
  took?: number
  Visual: () => React.ReactNode
}

/** One recorded run, walked through step by step. */
export interface Scenario {
  id: string
  /** The kind of input, e.g. "Before and after". */
  input: string
  /** The fine-tuned model it showcases. */
  model: string
  /** One sentence on what was recorded. */
  summary: string
  thumbnail: string
  steps: Array<DemoStep>
}
