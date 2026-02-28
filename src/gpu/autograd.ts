import { Tensor, type GradFn } from './tensor'

interface TapeEntry {
  output: Tensor
  parents: Tensor[]
  gradFn: GradFn
}

let tape: TapeEntry[] = []
let gradEnabled = true

export function recordOp(output: Tensor, parents: Tensor[], gradFn: GradFn): void {
  if (!gradEnabled) return
  output._gradFn = gradFn
  output._parents = parents
  tape.push({ output, parents, gradFn })
}

export async function backward(loss: Tensor): Promise<void> {
  // Initialize loss gradient to 1
  loss.grad = await Tensor.ones(loss.shape)

  // Walk tape in reverse
  for (let i = tape.length - 1; i >= 0; i--) {
    const entry = tape[i]
    if (entry.output.grad && entry.gradFn) {
      await entry.gradFn(entry.output.grad)
    }
  }
}

export function clearTape(): void {
  tape = []
}

export function zeroGrad(params: Tensor[]): void {
  for (const p of params) {
    if (p.grad) {
      p.grad.dispose()
      p.grad = null
    }
  }
}

export function noGrad<T>(fn: () => T): T {
  const prev = gradEnabled
  gradEnabled = false
  try {
    return fn()
  } finally {
    gradEnabled = prev
  }
}

export function isGradEnabled(): boolean {
  return gradEnabled
}

export function setGradEnabled(enabled: boolean): void {
  gradEnabled = enabled
}
