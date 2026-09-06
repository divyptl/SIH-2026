import * as React from 'react'
import { CircleAlertIcon, InfoIcon } from 'lucide-react'

import { Alert, AlertDescription, AlertTitle } from '#/components/ui/alert'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '#/components/ui/accordion'
import { Badge } from '#/components/ui/badge'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '#/components/ui/card'
import {
  Progress,
  ProgressLabel,
  ProgressValue,
} from '#/components/ui/progress'
import { Separator } from '#/components/ui/separator'
import { EvidenceOverlay } from '#/components/evidence-overlay'
import { CONFIGURATION_LABELS, TASK_LABELS } from '#/lib/api'
import type { AnalysisResponse } from '#/lib/api'

interface AnalysisResultProps {
  result: AnalysisResponse
}

function formatMs(ms: number) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`
}

export function AnalysisResult({ result }: AnalysisResultProps) {
  const [activeEvidence, setActiveEvidence] = React.useState<number | null>(
    null,
  )

  const confidencePercent = Math.round(result.confidence * 100)
  const observations = result.evidence.filter(
    (item) => item.type === 'observation',
  )
  // The server renders every input to a browser-displayable JPEG; anything
  // without one cannot be shown (and never has boxes drawn on it).
  const previews = result.inputs.filter((info) => info.preview_data_uri)

  return (
    <Card>
      <CardHeader>
        <CardTitle>Analysis</CardTitle>
        <CardDescription>
          {TASK_LABELS[result.task]} &middot;{' '}
          {CONFIGURATION_LABELS[result.trace.input_configuration]} &middot;{' '}
          {formatMs(result.execution_time_ms)}
        </CardDescription>
      </CardHeader>

      <CardContent className="flex flex-col gap-4">
        {!result.trace.domain_adapted && (
          <Alert>
            <InfoIcon />
            <AlertTitle>Generic baseline, not domain-adapted</AlertTitle>
            <AlertDescription>
              No fine-tuned remote-sensing specialist is registered for this
              task yet, so <code>{result.model_name}</code> answered it. Treat
              this as a baseline result.
            </AlertDescription>
          </Alert>
        )}

        <p className="text-sm/relaxed whitespace-pre-line">{result.answer}</p>

        <Progress value={confidencePercent} className="gap-1.5">
          <ProgressLabel className="text-xs text-muted-foreground">
            Confidence
          </ProgressLabel>
          <ProgressValue className="ml-auto text-xs tabular-nums" />
        </Progress>

        {previews.length > 0 && (
          <>
            <Separator />
            <div
              className={
                previews.length > 1
                  ? 'grid gap-3 sm:grid-cols-2'
                  : 'flex flex-col gap-3'
              }
            >
              {previews.map((info) => {
                const forThisImage = result.evidence.filter(
                  (item) => item.image_index === info.index,
                )
                return (
                  <div key={info.index} className="flex flex-col gap-1.5">
                    <EvidenceOverlay
                      src={info.preview_data_uri!}
                      alt={info.filename}
                      evidence={forThisImage}
                      activeIndex={
                        activeEvidence !== null
                          ? forThisImage.indexOf(
                              result.evidence[activeEvidence],
                            )
                          : null
                      }
                      onHoverChange={(local) =>
                        setActiveEvidence(
                          local === null
                            ? null
                            : result.evidence.indexOf(forThisImage[local]),
                        )
                      }
                    />
                    <div className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
                      {result.inputs.length > 1 && (
                        <Badge variant="secondary">Image {info.index}</Badge>
                      )}
                      <Badge variant="secondary">{info.modality}</Badge>
                      <span>{info.detected_format}</span>
                      <span>
                        {info.width}&times;{info.height}
                      </span>
                      {info.is_georeferenced && <Badge>georeferenced</Badge>}
                    </div>
                  </div>
                )
              })}
            </div>
          </>
        )}

        {result.evidence.length > 0 && (
          <div className="flex flex-col gap-2">
            <h3 className="text-sm font-medium">
              Evidence ({result.evidence.length})
            </h3>
            <ul className="flex flex-col gap-1.5">
              {result.evidence.map((item, index) => (
                <li
                  key={index}
                  onMouseEnter={() => setActiveEvidence(index)}
                  onMouseLeave={() => setActiveEvidence(null)}
                  className={
                    'flex items-start gap-2 rounded-lg border p-2 text-sm transition-colors ' +
                    (activeEvidence === index ? 'bg-muted' : '')
                  }
                >
                  <Badge
                    variant={item.type === 'bbox' ? 'default' : 'secondary'}
                  >
                    {item.type}
                  </Badge>
                  <div className="flex min-w-0 flex-col gap-0.5">
                    {item.label && (
                      <span className="font-medium">{item.label}</span>
                    )}
                    <span className="text-muted-foreground">
                      {item.description}
                    </span>
                  </div>
                  {item.confidence !== null && (
                    <span className="ml-auto shrink-0 text-xs tabular-nums text-muted-foreground">
                      {Math.round(item.confidence * 100)}%
                    </span>
                  )}
                </li>
              ))}
            </ul>
            {observations.length > 0 && previews.length > 0 && (
              <p className="text-xs text-muted-foreground">
                {observations.length} item(s) have no spatial extent and are not
                drawn on the imagery.
              </p>
            )}
          </div>
        )}

        {result.trace.warnings.length > 0 && (
          <Alert>
            <CircleAlertIcon />
            <AlertTitle>Notes from the controller</AlertTitle>
            <AlertDescription>
              <ul className="ml-4 flex list-disc flex-col gap-1">
                {result.trace.warnings.map((warning, index) => (
                  <li key={index}>{warning}</li>
                ))}
              </ul>
            </AlertDescription>
          </Alert>
        )}

        <Accordion>
          <AccordionItem value="trace">
            <AccordionTrigger>
              Execution trace ({result.trace.steps.length} steps)
            </AccordionTrigger>
            <AccordionContent>
              <div className="flex flex-col gap-3">
                <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
                  <dt className="text-muted-foreground">Request</dt>
                  <dd className="font-mono">{result.request_id}</dd>
                  <dt className="text-muted-foreground">Routing</dt>
                  <dd>
                    {result.trace.task} ({result.trace.task_source})
                    {result.trace.routing_rationale
                      ? ` — ${result.trace.routing_rationale}`
                      : ''}
                  </dd>
                  <dt className="text-muted-foreground">Tools</dt>
                  <dd>{result.trace.selected_tools.join(', ')}</dd>
                  {result.usage?.total_tokens != null && (
                    <>
                      <dt className="text-muted-foreground">Tokens</dt>
                      <dd className="tabular-nums">
                        {result.usage.total_tokens}
                      </dd>
                    </>
                  )}
                </dl>

                <ol className="flex flex-col gap-2">
                  {result.trace.steps.map((step, index) => (
                    <li key={index} className="rounded-lg border p-2 text-xs">
                      <div className="flex items-center gap-2">
                        <Badge variant="secondary">{step.stage}</Badge>
                        <span className="font-medium">{step.tool}</span>
                        <span className="ml-auto tabular-nums text-muted-foreground">
                          {formatMs(step.duration_ms)}
                        </span>
                      </div>
                      {step.detail && (
                        <p className="mt-1 text-muted-foreground">
                          {step.detail}
                        </p>
                      )}
                      {step.model && (
                        <p className="mt-0.5 font-mono text-[11px] text-muted-foreground">
                          {step.model}
                        </p>
                      )}
                    </li>
                  ))}
                </ol>
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </CardContent>
    </Card>
  )
}
