import type * as React from 'react'
import { useRender } from '@base-ui/react/use-render'
import { cva } from 'class-variance-authority'
import type { VariantProps } from 'class-variance-authority'
import { cn } from 'cn'

const typographyVariants = cva('', {
  variants: {
    variant: {
      h1: 'scroll-m-20 font-heading text-4xl font-extrabold tracking-tight text-balance',
      h2: 'scroll-m-20 border-b pb-2 font-heading text-3xl font-semibold tracking-tight first:mt-0',
      h3: 'scroll-m-20 font-heading text-2xl font-semibold tracking-tight',
      h4: 'scroll-m-20 font-heading text-xl font-semibold tracking-tight',
      p: 'leading-7 [&:not(:first-child)]:mt-6',
      blockquote: 'mt-6 border-l-2 pl-6 italic',
      list: 'my-6 ml-6 list-disc [&>li]:mt-2',
      code: 'relative rounded bg-muted px-[0.3rem] py-[0.2rem] font-mono text-sm font-semibold',
      lead: 'text-xl text-muted-foreground',
      large: 'text-lg font-semibold',
      small: 'text-sm leading-none font-medium',
      muted: 'text-sm text-muted-foreground',
    },
  },
  defaultVariants: {
    variant: 'p',
  },
})

type TypographyVariant = NonNullable<
  VariantProps<typeof typographyVariants>['variant']
>

const typographyTags = {
  h1: 'h1',
  h2: 'h2',
  h3: 'h3',
  h4: 'h4',
  p: 'p',
  blockquote: 'blockquote',
  list: 'ul',
  code: 'code',
  lead: 'p',
  large: 'div',
  small: 'small',
  muted: 'p',
} as const satisfies Record<
  TypographyVariant,
  keyof React.JSX.IntrinsicElements
>

function Typography({
  className,
  variant = 'p',
  render,
  ...props
}: useRender.ComponentProps<'p'> & VariantProps<typeof typographyVariants>) {
  const resolvedVariant = variant ?? 'p'

  return useRender({
    render,
    defaultTagName: typographyTags[resolvedVariant],
    props: {
      'data-slot': 'typography',
      'data-variant': resolvedVariant,
      className: cn(
        typographyVariants({ variant: resolvedVariant, className }),
      ),
      ...props,
    },
  })
}

export { Typography, typographyVariants }
export type { TypographyVariant }
