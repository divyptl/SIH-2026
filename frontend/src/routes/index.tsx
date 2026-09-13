import * as React from 'react'
import { createFileRoute } from '@tanstack/react-router'
import {
  CircleAlertIcon,
  FileImageIcon,
  ImageUpIcon,
  PaperclipIcon,
  ScanSearchIcon,
  XIcon,
} from 'lucide-react'
import { cn } from 'cn'

import { AnalysisResult } from '#/components/analysis-result'
import { Alert, AlertDescription, AlertTitle } from '#/components/ui/alert'
import { Badge } from '#/components/ui/badge'
import { Button } from '#/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '#/components/ui/card'
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from '#/components/ui/empty'
import {
  Field,
  FieldDescription,
  FieldError,
  FieldGroup,
  FieldLabel,
} from '#/components/ui/field'
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupText,
  InputGroupTextarea,
} from '#/components/ui/input-group'
import { Skeleton } from '#/components/ui/skeleton'
import { Spinner } from '#/components/ui/spinner'
import { ApiError, analyse } from '#/lib/api'
import type { AnalysisResponse } from '#/lib/api'

export const Route = createFileRoute('/')({ component: Home })

const ACCEPTED_TYPES = ['image/tiff', 'image/x-tiff']
const ACCEPTED_EXTENSIONS = ['.tif', '.tiff']
// The picker needs both: some systems report no MIME type at all for GeoTIFF.
const ACCEPT_ATTRIBUTE = [...ACCEPTED_TYPES, ...ACCEPTED_EXTENSIONS].join(',')
const MAX_FILE_SIZE = 20 * 1024 * 1024
const MAX_IMAGES = 2

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

/** Accept on MIME type, falling back to extension when the type is missing. */
function isAcceptedImage(file: File) {
  if (ACCEPTED_TYPES.includes(file.type)) return true
  const name = file.name.toLowerCase()
  return ACCEPTED_EXTENSIONS.some((extension) => name.endsWith(extension))
}

/** Short format label for the preview badge. */
function formatLabel(_file: File) {
  return 'GEOTIFF'
}

function Home() {
  const inputRef = React.useRef<HTMLInputElement>(null)
  const dragDepth = React.useRef(0)

  const [files, setFiles] = React.useState<Array<File>>([])
  const [previewUrls, setPreviewUrls] = React.useState<Array<string>>([])
  const [prompt, setPrompt] = React.useState('')
  const [isDragging, setIsDragging] = React.useState(false)
  const [failedThumbnails, setFailedThumbnails] = React.useState<Set<number>>(
    new Set(),
  )
  const [error, setError] = React.useState<string | null>(null)

  const abortRef = React.useRef<AbortController | null>(null)
  const [isAnalysing, setIsAnalysing] = React.useState(false)
  const [result, setResult] = React.useState<AnalysisResponse | null>(null)
  const [apiError, setApiError] = React.useState<string | null>(null)

  // Keep the object URLs in sync with the selected files and release them on swap.
  React.useEffect(() => {
    if (files.length === 0) {
      setPreviewUrls([])
      return
    }
    const urls = files.map((f) => URL.createObjectURL(f))
    setPreviewUrls(urls)
    return () => urls.forEach((url) => URL.revokeObjectURL(url))
  }, [files])

  // Drop any in-flight request if the user navigates away mid-analysis.
  React.useEffect(() => () => abortRef.current?.abort(), [])

  const addFiles = (candidates: Array<File>) => {
    if (candidates.length === 0) return
    const room = MAX_IMAGES - files.length
    if (room <= 0) {
      setError(
        `Up to ${MAX_IMAGES} images per analysis (a cross-modal or bi-temporal pair). Remove one first.`,
      )
      return
    }
    if (candidates.length > room) {
      setError(
        `Up to ${MAX_IMAGES} images per analysis. Only the first ${room} of your selection were added.`,
      )
      candidates = candidates.slice(0, room)
    }

    for (const candidate of candidates) {
      if (!isAcceptedImage(candidate)) {
        setError('Unsupported format. Upload a GeoTIFF (.tif/.tiff).')
        return
      }
      if (candidate.size > MAX_FILE_SIZE) {
        setError(
          `Image is too large. Keep it under ${formatBytes(MAX_FILE_SIZE)}.`,
        )
        return
      }
    }

    setError(null)
    setFailedThumbnails(new Set())
    setFiles((prev) => [...prev, ...candidates])
  }

  const removeFile = (index: number) => {
    setError(null)
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const clearFiles = () => {
    setFiles([])
    setError(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  // Depth counter so dragging over child nodes doesn't flicker the highlight.
  const onDragEnter = (e: React.DragEvent) => {
    e.preventDefault()
    dragDepth.current += 1
    if (e.dataTransfer.types.includes('Files')) setIsDragging(true)
  }

  const onDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    dragDepth.current -= 1
    if (dragDepth.current <= 0) setIsDragging(false)
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    dragDepth.current = 0
    setIsDragging(false)
    addFiles(Array.from(e.dataTransfer.files))
  }

  const handleSubmit = async (e: React.SyntheticEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (files.length === 0) {
      setError('At least one image is required to run an analysis.')
      return
    }
    if (!prompt.trim() || isAnalysing) return

    // Abort any in-flight request so a fast resubmit can't race an older one.
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setIsAnalysing(true)
    setError(null)
    setApiError(null)
    setResult(null)

    try {
      const response = await analyse({
        prompt: prompt.trim(),
        images: files,
        signal: controller.signal,
      })
      setResult(response)
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === 'AbortError') return
      setApiError(
        cause instanceof ApiError
          ? cause.message
          : 'Something went wrong running the analysis.',
      )
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null
        setIsAnalysing(false)
      }
    }
  }

  const canSubmit =
    files.length > 0 && prompt.trim().length > 0 && !isAnalysing

  return (
    <div className="flex justify-center p-4 sm:p-8">
      <form onSubmit={handleSubmit} className="w-full max-w-2xl">
        <Card>
          <CardHeader>
            <CardTitle>Let&rsquo;s analyze your image!</CardTitle>
            <CardDescription>
              Upload a SAR or optical capture and ask anything about what it
              shows.
            </CardDescription>
          </CardHeader>

          <CardContent>
            <FieldGroup>
              <Field data-invalid={error ? true : undefined}>
                <FieldLabel htmlFor="image">
                  Image(s)
                  <span aria-hidden className="text-destructive">
                    *
                  </span>
                </FieldLabel>

                <input
                  ref={inputRef}
                  id="image"
                  name="image"
                  type="file"
                  accept={ACCEPT_ATTRIBUTE}
                  multiple={files.length < MAX_IMAGES - 1}
                  aria-required
                  className="sr-only"
                  onChange={(e) => {
                    addFiles(Array.from(e.target.files ?? []))
                    e.target.value = ''
                  }}
                />

                {files.length > 0 ? (
                  <div className="flex flex-col gap-2">
                    {files.map((f, index) => (
                      <div
                        key={`${f.name}-${index}`}
                        className="flex items-center gap-3 rounded-xl border p-2"
                      >
                        {failedThumbnails.has(index) ? (
                          // Browsers cannot decode TIFF/GeoTIFF, so fall back
                          // to an icon rather than showing a broken image.
                          <div className="flex size-16 shrink-0 items-center justify-center rounded-lg bg-muted text-muted-foreground">
                            <FileImageIcon />
                          </div>
                        ) : (
                          <img
                            src={previewUrls[index]}
                            alt={`Preview of ${f.name}`}
                            onError={() =>
                              setFailedThumbnails(
                                (prev) => new Set(prev).add(index),
                              )
                            }
                            className="size-16 shrink-0 rounded-lg object-cover"
                          />
                        )}
                        <div className="flex min-w-0 flex-col gap-1">
                          <span className="truncate text-sm font-medium">
                            {f.name}
                          </span>
                          <div className="flex items-center gap-1.5">
                            <Badge variant="secondary">
                              {formatLabel(f)}
                            </Badge>
                            <span className="text-xs text-muted-foreground">
                              {formatBytes(f.size)}
                            </span>
                          </div>
                        </div>
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon-sm"
                          aria-label={`Remove ${f.name}`}
                          className="ml-auto"
                          onClick={() => removeFile(index)}
                        >
                          <XIcon />
                        </Button>
                      </div>
                    ))}
                    {files.length < MAX_IMAGES ? (
                      <div
                        onDragEnter={onDragEnter}
                        onDragOver={(e) => e.preventDefault()}
                        onDragLeave={onDragLeave}
                        onDrop={onDrop}
                        onClick={() => inputRef.current?.click()}
                        className={cn(
                          'cursor-pointer rounded-xl border border-dashed py-3 text-center text-sm text-muted-foreground transition-colors hover:bg-muted/50',
                          isDragging && 'border-ring bg-muted/50',
                        )}
                      >
                        {isDragging
                          ? 'Drop to attach'
                          : 'Add a second image (optical+SAR pair or before/after)'}
                      </div>
                    ) : null}
                  </div>
                ) : (
                  <div
                    onDragEnter={onDragEnter}
                    onDragOver={(e) => e.preventDefault()}
                    onDragLeave={onDragLeave}
                    onDrop={onDrop}
                    onClick={() => inputRef.current?.click()}
                    className={cn(
                      'cursor-pointer rounded-xl border border-dashed transition-colors hover:bg-muted/50',
                      isDragging && 'border-ring bg-muted/50',
                    )}
                  >
                    <Empty className="border-0 py-8">
                      <EmptyHeader>
                        <EmptyMedia variant="icon">
                          <ImageUpIcon />
                        </EmptyMedia>
                        <EmptyTitle>
                          {isDragging
                            ? 'Drop to attach'
                            : 'Drop image(s) here'}
                        </EmptyTitle>
                        <EmptyDescription>
                          or click to browse &mdash; one image, or two for a
                          cross-modal / bi-temporal pair
                        </EmptyDescription>
                      </EmptyHeader>
                    </Empty>
                  </div>
                )}

                <FieldDescription>
                  Required. One georeferenced GeoTIFF (.tif/.tiff), or two for a
                  co-registered optical+SAR pair or before/after comparison
                  &mdash; each up to {formatBytes(MAX_FILE_SIZE)}.
                </FieldDescription>
                {error ? <FieldError>{error}</FieldError> : null}
              </Field>

              <Field>
                <FieldLabel htmlFor="prompt">
                  Prompt
                  <span aria-hidden className="text-destructive">
                    *
                  </span>
                </FieldLabel>
                <InputGroup>
                  <InputGroupTextarea
                    id="prompt"
                    name="prompt"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="e.g. How many ships are docked in the harbour?"
                    aria-required
                    className="min-h-20"
                  />
                  <InputGroupAddon align="block-end">
                    <InputGroupButton
                      type="button"
                      disabled={files.length >= MAX_IMAGES}
                      onClick={() => inputRef.current?.click()}
                    >
                      <PaperclipIcon />
                      {files.length > 0 ? 'Add image' : 'Attach image'}
                    </InputGroupButton>
                    <InputGroupText className="ml-auto text-xs">
                      {files.length === 0
                        ? 'No attachment'
                        : `${files.length} attachment${files.length > 1 ? 's' : ''}`}
                    </InputGroupText>
                  </InputGroupAddon>
                </InputGroup>
              </Field>
            </FieldGroup>
          </CardContent>

          <CardFooter className="justify-end gap-2">
            <Button
              type="button"
              variant="ghost"
              disabled={(files.length === 0 && !prompt) || isAnalysing}
              onClick={() => {
                clearFiles()
                setPrompt('')
                setResult(null)
                setApiError(null)
              }}
            >
              Reset
            </Button>
            <Button type="submit" disabled={!canSubmit}>
              {isAnalysing ? (
                <Spinner data-icon="inline-start" />
              ) : (
                <ScanSearchIcon data-icon="inline-start" />
              )}
              {isAnalysing ? 'Analyzing…' : 'Analyze'}
            </Button>
          </CardFooter>
        </Card>

        {apiError && (
          <Alert variant="destructive" className="mt-4">
            <CircleAlertIcon />
            <AlertTitle>Analysis failed</AlertTitle>
            <AlertDescription>{apiError}</AlertDescription>
          </Alert>
        )}

        {isAnalysing && (
          <Card className="mt-4">
            <CardContent className="flex flex-col gap-2.5 py-2">
              <Skeleton className="h-4 w-2/5" />
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-4/5" />
              <p className="text-xs text-muted-foreground">
                Routing the query and running the selected specialist…
              </p>
            </CardContent>
          </Card>
        )}

        {result && !isAnalysing && (
          <div className="mt-4">
            <AnalysisResult result={result} />
          </div>
        )}
      </form>
    </div>
  )
}
