export { cn } from 'cn'

/** The value of a single-thumb slider, which reports one value per thumb. */
export function firstValue(value: number | ReadonlyArray<number>) {
  return typeof value === 'number' ? value : value[0]
}
