import * as React from 'react'
import { createFileRoute } from '@tanstack/react-router'
import {
  CircleAlertIcon,
  PaperclipIcon,
  RotateCcwIcon,
  ScanSearchIcon,
} from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { useTranslation } from 'react-i18next'

import { AnalysisPending } from '#/components/analysis-pending'
import { AnalysisResult } from '#/components/analysis-result'
import { EmptySlot, FilledSlot, formatBytes } from '#/components/image-slot'
import { PromptSuggestions } from '#/components/prompt-suggestions'
import { Hero } from '#/components/hero'
import { ResultEmpty } from '#/components/result-empty'
import { Alert, AlertDescription, AlertTitle } from '#/components/ui/alert'
import { Button } from '#/components/ui/button'
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

const ACCEPTED_TYPES = ['image/tiff', 'image/x-tiff', 'image/png', 'image/jpeg']
const ACCEPTED_EXTENSIONS = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
// The picker needs both: some systems report no MIME type at all for GeoTIFF.
const ACCEPT_ATTRIBUTE = [...ACCEPTED_TYPES, ...ACCEPTED_EXTENSIONS].join(',')

const MAX_FILE_SIZE = 20 * 1024 * 1024
const MAX_IMAGES = 2

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
    // The result sits below the input panel; bring it into view.
    resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })

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

      <div className="mx-auto flex max-w-6xl flex-col gap-12 px-4 pb-20 sm:px-6">
        <form
          onSubmit={handleSubmit}
          className="grid overflow-hidden rounded-xl bg-card ring-1 ring-foreground/10 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]"
        >
          <Field
            data-invalid={error ? true : undefined}
            className="gap-3 p-4 sm:p-5"
          >
            <FieldLabel htmlFor="image" className="font-heading text-base">
              {t('home.imagesLabel')}
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

            <div className="grid grid-cols-2 gap-3">
              {Array.from({ length: MAX_IMAGES }, (_, slot) => {
                const file = files.at(slot)
                return (
                  <AnimatePresence key={slot} mode="popLayout" initial={false}>
                    <motion.div
                      key={
                        file
                          ? `${file.name}-${file.size}-${file.lastModified}`
                          : 'empty'
                      }
                      initial={{ opacity: 0, scale: 0.97 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.97 }}
                      transition={{ duration: 0.18 }}
                    >
                      {file ? (
                        <FilledSlot
                          file={file}
                          // Browsers cannot show GeoTIFF; after an analysis the
                          // server's rendering of it can stand in.
                          serverPreview={
                            result?.inputs.find(
                              (info) => info.filename === file.name,
                            )?.preview_data_uri
                          }
                          onRemove={() => removeFile(slot)}
                        />
                      ) : (
                        <EmptySlot
                          primary={slot === 0}
                          isDragging={isDragging}
                          dropHandlers={dropHandlers}
                          onPick={() => inputRef.current?.click()}
                        />
                      )}
                    </motion.div>
                  </AnimatePresence>
                )
              })}
            </div>

            <FieldDescription className="text-xs/relaxed">
              {t('home.imagesHelp', { size: formatBytes(MAX_FILE_SIZE) })}
            </FieldDescription>
            {error ? <FieldError>{error}</FieldError> : null}
          </Field>

          <div className="flex flex-col gap-5 border-t p-4 sm:p-5 lg:border-s lg:border-t-0">
            <Field className="gap-3">
              <FieldLabel htmlFor="prompt" className="font-heading text-base">
                {t('home.promptLabel')}
              </FieldLabel>
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
                  className="min-h-28 text-base"
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

            <div className="mt-auto flex items-center gap-2 border-t pt-4">
              <Button
                type="button"
                variant="ghost"
                disabled={files.length === 0 && !prompt && !result}
                onClick={reset}
              >
                <RotateCcwIcon data-icon="inline-start" />
                {t('home.reset')}
              </Button>
              <Button
                type="submit"
                size="lg"
                disabled={!canSubmit}
                className="ms-auto px-5"
              >
                {isAnalysing ? (
                  <Spinner data-icon="inline-start" />
                ) : (
                  <ScanSearchIcon data-icon="inline-start" />
                )}
                {isAnalysing ? t('home.analyzing') : t('home.analyze')}
              </Button>
            </div>
          </div>
        </form>

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
                <AnalysisPending files={files} />
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
