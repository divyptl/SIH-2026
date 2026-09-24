import { ScanSearchIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from '#/components/ui/empty'

/** The result area before the first analysis. */
export function ResultEmpty() {
  const { t } = useTranslation()

  return (
    <Empty className="min-h-80 border border-dashed">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <ScanSearchIcon />
        </EmptyMedia>
        <EmptyTitle>{t('result.emptyTitle')}</EmptyTitle>
        <EmptyDescription>{t('result.emptyBody')}</EmptyDescription>
      </EmptyHeader>
    </Empty>
  )
}
