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

export interface Evidence {
  type: 'bbox' | 'mask' | 'heatmap' | 'observation'
  data: BoundingBox | null
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
  notes: Array<string>
  /** Server-rendered JPEG of exactly what the model was shown. */
  preview_data_uri: string | null
}

export interface TraceStep {
  stage: 'validate' | 'classify' | 'select' | 'execute' | 'aggregate'
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
}

export const TASK_LABELS: Record<Task, string> = {
  vqa: 'Visual question answering',
  caption: 'Scene description',
  grounding: 'Region grounding',
  change_vqa: 'Change VQA',
  change_description: 'Change description',
  fusion: 'Optical–SAR fusion',
}

export const CONFIGURATION_LABELS: Record<InputConfiguration, string> = {
  single: 'Single image',
  cross_modal_pair: 'Cross-modal pair',
  bi_temporal_pair: 'Bi-temporal pair',
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
  signal?: AbortSignal
}

export async function analyse({
  prompt,
  images,
  modalities,
  task,
  signal,
}: AnalyseArgs): Promise<AnalysisResponse> {
  const body = new FormData()
  body.append('prompt', prompt)
  for (const image of images) body.append('images', image)
  if (modalities?.length) body.append('modalities', modalities.join(','))
  if (task) body.append('task', task)

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
