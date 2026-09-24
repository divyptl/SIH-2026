import { useReducedMotion } from 'motion/react'
import { useTranslation } from 'react-i18next'

import ShinyText from '#/components/ShinyText'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '#/components/ui/card'
import { Skeleton } from '#/components/ui/skeleton'

/** Placeholder for the result while the request is in flight. */
export function AnalysisPending({ fileCount }: { fileCount: number }) {
  const { t } = useTranslation()
  const reduceMotion = useReducedMotion()
  const tiles = Math.max(1, fileCount)

  return (
    <Card>
      <CardHeader>
        <CardTitle>
          {reduceMotion ? (
            t('home.analyzing')
          ) : (
            <ShinyText
              text={t('home.analyzing')}
              color="var(--muted-foreground)"
              shineColor="var(--foreground)"
              speed={2}
            />
          )}
        </CardTitle>
        <CardDescription>{t('home.running')}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div className={tiles > 1 ? 'grid gap-3 sm:grid-cols-2' : 'grid'}>
          {Array.from({ length: tiles }, (_, index) => (
            <Skeleton key={index} className="aspect-[4/3] w-full" />
          ))}
        </div>
        <div className="flex flex-col gap-2">
          <Skeleton className="h-4 w-2/5" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-4/5" />
        </div>
      </CardContent>
    </Card>
  )
}
