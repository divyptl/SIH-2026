import * as React from 'react'
import { createFileRoute } from '@tanstack/react-router'
import {
  CircleAlertIcon,
  FileImageIcon,
  ImageUpIcon,
  PaperclipIcon,
  PlusIcon,
  RotateCcwIcon,
  ScanSearchIcon,
  XIcon,
} from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { cn } from 'cn'
import { useTranslation } from 'react-i18next'

import { AnalysisPending } from '#/components/analysis-pending'
import { AnalysisResult } from '#/components/analysis-result'
import { PromptSuggestions } from '#/components/prompt-suggestions'
import { Hero } from '#/components/hero'
import { ResultEmpty } from '#/components/result-empty'
import { Alert, AlertDescription, AlertTitle } from '#/components/ui/alert'
import { Badge } from '#/components/ui/badge'
import { Button } from '#/components/ui/button'
import { Card } from '#/components/ui/card'
import {
  Field,
  FieldDescription,
  FieldError,
  FieldLabel,
} from '#/components/ui/field'
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupText,
  InputGroupTextarea,
} from '#/components/ui/input-group'
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

function Home() {
  const { t, i18n } = useTranslation()
  const inputRef = React.useRef<HTMLInputElement>(null)
  const promptRef = React.useRef<HTMLTextAreaElement>(null)
  const resultRef = React.useRef<HTMLDivElement>(null)
  const dragDepth = React.useRef(0)

  const [files, setFiles] = React.useState<Array<File>>([])
  const [prompt, setPrompt] = React.useState('')
  const [isDragging, setIsDragging] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  const abortRef = React.useRef<AbortController | null>(null)
  const [isAnalysing, setIsAnalysing] = React.useState(false)
  const [result, setResult] = React.useState<AnalysisResponse | null>(null)
  // The question as submitted; the prompt box may be edited after the fact.
  const [resultQuery, setResultQuery] = React.useState('')
  const [apiError, setApiError] = React.useState<string | null>(null)

  // Drop any in-flight request if the user navigates away mid-analysis.
  React.useEffect(() => () => abortRef.current?.abort(), [])

  const addFiles = (candidates: Array<File>) => {
    if (candidates.length === 0) return
    const room = MAX_IMAGES - files.length
    if (room <= 0) {
      setError(t('errors.tooManyFull', { max: MAX_IMAGES }))
      return
    }
    let partial = false
    if (candidates.length > room) {
      partial = true
      candidates = candidates.slice(0, room)
    }

    for (const candidate of candidates) {
      if (!isAcceptedImage(candidate)) {
        setError(t('errors.unsupported'))
        return
      }
      if (candidate.size > MAX_FILE_SIZE) {
        setError(t('errors.tooLarge', { size: formatBytes(MAX_FILE_SIZE) }))
        return
      }
    }

    setError(partial ? t('errors.tooManyPartial', { max: MAX_IMAGES }) : null)
    setFiles((prev) => [...prev, ...candidates])
  }

  const removeFile = (index: number) => {
    setError(null)
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const reset = () => {
    abortRef.current?.abort()
    setFiles([])
    setPrompt('')
    setError(null)
    setResult(null)
    setApiError(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  // Depth counter so dragging over child nodes doesn't flicker the highlight.
  const dropHandlers = {
    onDragEnter: (e: React.DragEvent) => {
      e.preventDefault()
      dragDepth.current += 1
      if (e.dataTransfer.types.includes('Files')) setIsDragging(true)
    },
    onDragOver: (e: React.DragEvent) => e.preventDefault(),
    onDragLeave: (e: React.DragEvent) => {
      e.preventDefault()
      dragDepth.current -= 1
      if (dragDepth.current <= 0) setIsDragging(false)
    },
    onDrop: (e: React.DragEvent) => {
      e.preventDefault()
      dragDepth.current = 0
      setIsDragging(false)
      addFiles(Array.from(e.dataTransfer.files))
    },
  }

  const handleSubmit = async (e: React.SyntheticEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (files.length === 0) {
      setError(t('errors.imageRequired'))
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
    // On narrow screens the result panel sits below the form.
    if (window.matchMedia('(max-width: 1023px)').matches) {
      resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }

    const query = prompt.trim()
    try {
      const response = await analyse({
        prompt: query,
        images: files,
        language: i18n.resolvedLanguage,
        signal: controller.signal,
      })
      setResult(response)
      setResultQuery(query)
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === 'AbortError') return
      setApiError(
        cause instanceof ApiError ? cause.message : t('home.genericError'),
      )
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null
        setIsAnalysing(false)
      }
    }
  }

  const canSubmit = files.length > 0 && prompt.trim().length > 0 && !isAnalysing

  const applyExample = (text: string) => {
    setPrompt(text)
    promptRef.current?.focus()
  }

  return (
    <main>
      <Hero />

      <div className="mx-auto grid max-w-6xl gap-6 px-4 pb-16 sm:px-6 lg:grid-cols-[minmax(0,24rem)_minmax(0,1fr)] lg:items-start lg:gap-8">
        <Card className="lg:sticky lg:top-20">
          <form
            onSubmit={handleSubmit}
            className="flex flex-col gap-6 px-(--card-spacing)"
          >
            <Field data-invalid={error ? true : undefined} className="gap-3">
              <FieldLabel htmlFor="image">{t('home.imagesLabel')}</FieldLabel>

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

              <AnimatePresence initial={false}>
                {files.map((f, index) => (
                  <motion.div
                    key={`${f.name}-${f.size}-${f.lastModified}`}
                    layout
                    initial={{ opacity: 0, y: -6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, x: 12 }}
                    transition={{ duration: 0.18 }}
                    className="flex items-center gap-3 rounded-lg border p-2"
                  >
                    <div className="flex size-9 shrink-0 items-center justify-center rounded-md bg-muted text-muted-foreground">
                      <FileImageIcon className="size-4" />
                    </div>
                    <div className="flex min-w-0 flex-col gap-0.5">
                      <span className="truncate text-sm font-medium" dir="auto">
                        {f.name}
                      </span>
                      <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                        <Badge
                          variant="secondary"
                          className="h-4 px-1.5 text-[10px]"
                        >
                          GeoTIFF
                        </Badge>
                        <span className="tabular-nums">
                          {formatBytes(f.size)}
                        </span>
                      </span>
                    </div>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon-sm"
                      aria-label={t('home.remove', { name: f.name })}
                      className="ms-auto"
                      onClick={() => removeFile(index)}
                    >
                      <XIcon />
                    </Button>
                  </motion.div>
                ))}
              </AnimatePresence>

              {files.length < MAX_IMAGES && (
                <button
                  type="button"
                  {...dropHandlers}
                  onClick={() => inputRef.current?.click()}
                  className={cn(
                    'flex w-full flex-col items-center justify-center gap-2 rounded-lg border border-dashed px-4 text-center transition-colors hover:bg-muted/50 focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none',
                    files.length === 0 ? 'py-8' : 'py-3',
                    isDragging && 'border-foreground bg-muted',
                  )}
                >
                  {files.length === 0 ? (
                    <>
                      <span className="flex size-9 items-center justify-center rounded-md bg-muted text-muted-foreground">
                        <ImageUpIcon className="size-4" />
                      </span>
                      <span className="text-sm font-medium">
                        {isDragging
                          ? t('home.dropToAttach')
                          : t('home.dropHere')}
                      </span>
                      <span className="max-w-xs text-xs/relaxed text-muted-foreground">
                        {t('home.browseHint')}
                      </span>
                    </>
                  ) : (
                    <span className="flex items-center gap-2 text-sm text-muted-foreground">
                      <PlusIcon className="size-4 shrink-0" />
                      {isDragging
                        ? t('home.dropToAttach')
                        : t('home.addSecond')}
                    </span>
                  )}
                </button>
              )}

              <FieldDescription className="text-xs/relaxed">
                {t('home.imagesHelp', { size: formatBytes(MAX_FILE_SIZE) })}
              </FieldDescription>
              {error ? <FieldError>{error}</FieldError> : null}
            </Field>

            <Field className="gap-3">
              <FieldLabel htmlFor="prompt">{t('home.promptLabel')}</FieldLabel>
              <InputGroup>
                <InputGroupTextarea
                  ref={promptRef}
                  id="prompt"
                  name="prompt"
                  dir="auto"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  onKeyDown={(e) => {
                    // Ctrl/Cmd+Enter submits, the usual shortcut for multi-line inputs.
                    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                      e.preventDefault()
                      e.currentTarget.form?.requestSubmit()
                    }
                  }}
                  placeholder={t('home.promptPlaceholder')}
                  aria-required
                  aria-describedby="prompt-hint"
                  className="min-h-24"
                />
                <InputGroupAddon align="block-end">
                  <InputGroupButton
                    type="button"
                    disabled={files.length >= MAX_IMAGES}
                    onClick={() => inputRef.current?.click()}
                  >
                    <PaperclipIcon />
                    {files.length > 0
                      ? t('home.addImage')
                      : t('home.attachImage')}
                  </InputGroupButton>
                  <InputGroupText className="ms-auto text-xs tabular-nums">
                    {files.length === 0
                      ? t('home.noAttachment')
                      : t('home.attachments', { count: files.length })}
                  </InputGroupText>
                </InputGroupAddon>
              </InputGroup>

              <PromptSuggestions
                imageCount={files.length}
                onPick={applyExample}
              />

              <FieldDescription id="prompt-hint" className="text-xs/relaxed">
                {t('home.promptHint')}
              </FieldDescription>
            </Field>

            <div className="flex items-center gap-2 border-t pt-4">
              <Button
                type="button"
                variant="ghost"
                disabled={files.length === 0 && !prompt && !result}
                onClick={reset}
              >
                <RotateCcwIcon data-icon="inline-start" />
                {t('home.reset')}
              </Button>
              <Button type="submit" disabled={!canSubmit} className="ms-auto">
                {isAnalysing ? (
                  <Spinner data-icon="inline-start" />
                ) : (
                  <ScanSearchIcon data-icon="inline-start" />
                )}
                {isAnalysing ? t('home.analyzing') : t('home.analyze')}
              </Button>
            </div>
          </form>
        </Card>

        <section
          ref={resultRef}
          aria-live="polite"
          aria-busy={isAnalysing}
          className="min-w-0 scroll-mt-20"
        >
          <AnimatePresence mode="wait">
            {isAnalysing ? (
              <motion.div
                key="pending"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <AnalysisPending fileCount={files.length} />
              </motion.div>
            ) : apiError ? (
              <motion.div
                key="error"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
              >
                <Alert variant="destructive">
                  <CircleAlertIcon />
                  <AlertTitle>{t('home.analysisFailed')}</AlertTitle>
                  <AlertDescription>{apiError}</AlertDescription>
                </Alert>
              </motion.div>
            ) : result ? (
              <motion.div
                key={result.request_id}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
              >
                <AnalysisResult result={result} query={resultQuery} />
              </motion.div>
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <ResultEmpty />
              </motion.div>
            )}
          </AnimatePresence>
        </section>
      </div>
    </main>
  )
}
