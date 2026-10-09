/**
 * Client for the SatQuery AI backend.
 *
 * Types mirror `backend/schemas.py`; keep them in sync when the API changes.
 */

export const API_BASE_URL: string = (
  import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'
).replace(/\/$/, '')

export type Task =
  | 'vqa'
  | 'caption'
  | 'grounding'
  | 'change_vqa'
  | 'change_description'
  | 'fusion'

export type Modality = 'optical' | 'sar' | 'unknown'

export type InputConfiguration =
  'single' | 'cross_modal_pair' | 'bi_temporal_pair'

/** Normalised box in [0, 1], origin at the top-left of the image. */
export interface BoundingBox {
  x_min: number
  y_min: number
  x_max: number
  y_max: number
}

/** A change mask: a PNG data URI, opaque where the model predicts change. */
export interface ChangeMask {
  png: string
}

export interface Evidence {
  type: 'bbox' | 'mask' | 'heatmap' | 'observation'
  /** A box for `bbox`, a mask for `mask`, otherwise null. */
  data: BoundingBox | ChangeMask | null
  description: string
  label: string | null
  confidence: number | null
  image_index: number
}

export interface ImageInfo {
  index: number
  filename: string
  content_type: string | null
  detected_format: string
  modality: Modality
  width: number
  height: number
  size_bytes: number
  is_georeferenced: boolean
  /** Metres per pixel from the GeoTIFF tags; null when unknown. */
  ground_sample_distance_m: number | null
  notes: Array<string>
  /** Server-rendered JPEG of exactly what the model was shown. */
  preview_data_uri: string | null
}

export interface TraceStep {
  stage:
    | 'translate'
    | 'validate'
    | 'classify'
    | 'select'
    | 'execute'
    | 'narrate'
    | 'aggregate'
  tool: string
  model: string | null
  params: Record<string, unknown>
  detail: string
  duration_ms: number
}

export interface ExecutionTrace {
  input_configuration: InputConfiguration
  task: Task
  task_source: 'auto' | 'override'
  routing_rationale: string
  selected_tools: Array<string>
  steps: Array<TraceStep>
  total_duration_ms: number
  domain_adapted: boolean
  warnings: Array<string>
}

export interface Usage {
  prompt_tokens: number | null
  completion_tokens: number | null
  total_tokens: number | null
  cost: number | null
}

/** What the translation layer did; `answer`/`evidence` stay in English. */
export interface TranslationInfo {
  engine: string
  /** Language code the query was read as. */
  source_language: string
  /** True when inferred from the script rather than the language hint. */
  source_detected: boolean
  /** Language code the answer was translated to. */
  target_language: string
  original_query: string
  english_query: string
  /** Answer in `target_language`; null if back-translation failed. */
  answer: string | null
  /** Index-aligned with `evidence`; null if back-translation failed. */
  evidence_descriptions: Array<string> | null
  /** Index-aligned with `evidence`; null if back-translation failed. */
  evidence_labels?: Array<string | null> | null
  /** Index-aligned with `trace.warnings`; null if back-translation failed. */
  warnings?: Array<string> | null
  routing_rationale?: string | null
}

/**
 * A VLM's plain-language rewording of a specialist result. The wording is the
 * VLM's; every region and figure in it was checked against the specialist.
 */
export interface Narration {
  model: string
  /** The specialist's own answer, before rewording. */
  specialist_answer: string
  regions_described: number
}

export interface AnalysisResponse {
  request_id: string
  task: Task
  answer: string
  confidence: number
  evidence: Array<Evidence>
  model_name: string
  execution_time_ms: number
  inputs: Array<ImageInfo>
  trace: ExecutionTrace
  usage: Usage | null
  translation: TranslationInfo | null
  /** Set when a VLM reworded the specialist's result in plain language. */
  narration?: Narration | null
}

/** An error carrying the HTTP status, so callers can distinguish causes. */
export class ApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/**
 * FastAPI returns `detail` as a string for our raised errors, but as an array
 * of validation objects when request parsing itself fails.
 */
function extractDetail(payload: unknown, fallback: string): string {
  if (typeof payload !== 'object' || payload === null) return fallback
  const detail = (payload as { detail?: unknown }).detail

  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) =>
        typeof item === 'object' && item !== null && 'msg' in item
          ? String((item as { msg: unknown }).msg)
          : null,
      )
      .filter((msg): msg is string => Boolean(msg))
    if (messages.length > 0) return messages.join('; ')
  }
  return fallback
}

export interface AnalyseArgs {
  prompt: string
  /** One image, or two for a cross-modal or bi-temporal pair. */
  images: Array<File>
  /** Optional per-image modality hints, e.g. `['optical', 'sar']`. */
  modalities?: Array<Modality>
  /** Optional task override; omit to let the controller route. */
  task?: Task
  /**
   * The user's language code (see `src/lib/languages.ts`). Non-English queries
   * are translated to English server-side, and the answer is translated back.
   */
  language?: string
  signal?: AbortSignal
}

export async function analyse({
  prompt,
  images,
  modalities,
  task,
  language,
  signal,
}: AnalyseArgs): Promise<AnalysisResponse> {
  const body = new FormData()
  body.append('prompt', prompt)
  for (const image of images) body.append('images', image)
  if (modalities?.length) body.append('modalities', modalities.join(','))
  if (task) body.append('task', task)
  if (language) body.append('language', language)

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/api/analyse`, {
      method: 'POST',
      body,
      signal,
    })
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError')
      throw cause
    throw new ApiError(
      `Could not reach the SatQuery API at ${API_BASE_URL}. Is the backend running?`,
      0,
    )
  }

  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    throw new ApiError(
      extractDetail(payload, `Analysis failed (HTTP ${response.status}).`),
      response.status,
    )
  }

  return (await response.json()) as AnalysisResponse
}

/** Every string the PDF report prints, already in the report's language. */
export interface ReportLabels {
  title: string
  generated: string
  request: string
  question: string
  query_translated: string
  answer: string
  english_original: string
  task: string
  task_name: string
  input: string
  configuration_name: string
  confidence: string
  time: string
  tokens: string
  language: string
  language_name: string
  images: string
  /** Contains an `{{index}}` placeholder. */
  image: string
  modality: Record<Modality, string>
  georeferenced: string
  evidence: string
  /** Contains an `{{index}}` placeholder. */
  region: string
  no_spatial: string | null
  controller_notes: string
  trace: string
  routing: string
  tools: string
}

export interface ReportArgs {
  result: AnalysisResponse
  /** The question exactly as the user typed it. */
  query: string
  /** The report's language; `'en'` renders the English original only. */
  language: string
  labels: ReportLabels
}

/** Typeset a result into a PDF on the server and return it as a Blob. */
export async function downloadReport(args: ReportArgs): Promise<Blob> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/api/report`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(args),
    })
  } catch {
    throw new ApiError(
      `Could not reach the SatQuery API at ${API_BASE_URL}. Is the backend running?`,
      0,
    )
  }

  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    throw new ApiError(
      extractDetail(payload, `Report failed (HTTP ${response.status}).`),
      response.status,
    )
  }

  return response.blob()
}
