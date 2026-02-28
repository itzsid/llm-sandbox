import { getGPUDevice } from './device'
import { Tensor } from './tensor'
import { recordOp, isGradEnabled } from './autograd'

import elementwiseBinaryShader from './ops/elementwise_binary.wgsl'
import elementwiseUnaryShader from './ops/elementwise_unary.wgsl'
import matmulShader from './ops/matmul.wgsl'
import softmaxShader from './ops/softmax.wgsl'
import layernormShader from './ops/layernorm.wgsl'
import embeddingShader from './ops/embedding.wgsl'
import causalMaskShader from './ops/causal_mask.wgsl'
import crossentropyShader from './ops/crossentropy.wgsl'
import transposeShader from './ops/transpose.wgsl'
import reduceShader from './ops/reduce.wgsl'

// --- Pipeline cache ---
const pipelineCache = new Map<string, GPUComputePipeline>()

async function getPipeline(
  device: GPUDevice,
  shaderCode: string,
  entryPoint: string,
): Promise<GPUComputePipeline> {
  const key = shaderCode + '::' + entryPoint
  let pipeline = pipelineCache.get(key)
  if (!pipeline) {
    const module = device.createShaderModule({ code: shaderCode })
    pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint },
    })
    pipelineCache.set(key, pipeline)
  }
  return pipeline
}

// --- Uniform buffer helpers ---
function createUniformBuffer(device: GPUDevice, data: ArrayBuffer): GPUBuffer {
  const buf = device.createBuffer({
    size: Math.max(data.byteLength, 16), // min 16 bytes
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  })
  new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data))
  buf.unmap()
  return buf
}

function packU32(...values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length * 4)
  const view = new Uint32Array(buf)
  values.forEach((v, i) => (view[i] = v))
  return buf
}

function packMixed(specs: Array<{ type: 'u32' | 'f32'; value: number }>): ArrayBuffer {
  const buf = new ArrayBuffer(specs.length * 4)
  const u32 = new Uint32Array(buf)
  const f32 = new Float32Array(buf)
  specs.forEach((s, i) => {
    if (s.type === 'u32') u32[i] = s.value
    else f32[i] = s.value
  })
  return buf
}

// --- Dispatch helper ---
interface DispatchArgs {
  pipeline: GPUComputePipeline
  buffers: GPUBuffer[]
  uniformBuffer?: GPUBuffer
  workgroups: [number, number?, number?]
}

function dispatch(device: GPUDevice, args: DispatchArgs): void {
  const entries: GPUBindGroupEntry[] = args.buffers.map((b, i) => ({
    binding: i,
    resource: { buffer: b },
  }))
  if (args.uniformBuffer) {
    entries.push({
      binding: args.buffers.length,
      resource: { buffer: args.uniformBuffer },
    })
  }
  const bindGroup = device.createBindGroup({
    layout: args.pipeline.getBindGroupLayout(0),
    entries,
  })
  const encoder = device.createCommandEncoder()
  const pass = encoder.beginComputePass()
  pass.setPipeline(args.pipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(
    args.workgroups[0],
    args.workgroups[1] ?? 1,
    args.workgroups[2] ?? 1,
  )
  pass.end()
  device.queue.submit([encoder.finish()])
}

// --- Elementwise Binary ---
async function elementwiseBinary(a: Tensor, b: Tensor, opCode: number): Promise<Tensor> {
  if (a.size !== b.size) throw new Error(`Size mismatch: ${a.size} vs ${b.size}`)
  const device = await getGPUDevice()
  const result = await Tensor.empty(a.shape)
  const pipeline = await getPipeline(device, elementwiseBinaryShader, 'main')
  const uniform = createUniformBuffer(device, packU32(a.size, opCode))
  dispatch(device, {
    pipeline,
    buffers: [a.buffer, b.buffer, result.buffer],
    uniformBuffer: uniform,
    workgroups: [Math.ceil(a.size / 256)],
  })
  await device.queue.onSubmittedWorkDone()
  uniform.destroy()
  return result
}

export async function add(a: Tensor, b: Tensor): Promise<Tensor> {
  const result = await elementwiseBinary(a, b, 0)
  if (isGradEnabled() && (a.requiresGrad || b.requiresGrad)) {
    recordOp(result, [a, b], async (grad) => {
      if (a.requiresGrad) {
        if (a.grad) {
          const newGrad = await add(a.grad, grad)
          a.grad.dispose()
          a.grad = newGrad
        } else {
          a.grad = await copyTensor(grad)
        }
      }
      if (b.requiresGrad) {
        if (b.grad) {
          const newGrad = await add(b.grad, grad)
          b.grad.dispose()
          b.grad = newGrad
        } else {
          b.grad = await copyTensor(grad)
        }
      }
    })
  }
  return result
}

export async function sub(a: Tensor, b: Tensor): Promise<Tensor> {
  const result = await elementwiseBinary(a, b, 1)
  if (isGradEnabled() && (a.requiresGrad || b.requiresGrad)) {
    recordOp(result, [a, b], async (grad) => {
      if (a.requiresGrad) {
        if (a.grad) {
          const newGrad = await add(a.grad, grad)
          a.grad.dispose()
          a.grad = newGrad
        } else {
          a.grad = await copyTensor(grad)
        }
      }
      if (b.requiresGrad) {
        const negGrad = await neg(grad)
        if (b.grad) {
          const newGrad = await add(b.grad, negGrad)
          b.grad.dispose()
          negGrad.dispose()
          b.grad = newGrad
        } else {
          b.grad = negGrad
        }
      }
    })
  }
  return result
}

export async function mul(a: Tensor, b: Tensor): Promise<Tensor> {
  const result = await elementwiseBinary(a, b, 2)
  if (isGradEnabled() && (a.requiresGrad || b.requiresGrad)) {
    recordOp(result, [a, b], async (grad) => {
      if (a.requiresGrad) {
        const dA = await elementwiseBinary(grad, b, 2)
        if (a.grad) {
          const newGrad = await add(a.grad, dA)
          a.grad.dispose()
          dA.dispose()
          a.grad = newGrad
        } else {
          a.grad = dA
        }
      }
      if (b.requiresGrad) {
        const dB = await elementwiseBinary(grad, a, 2)
        if (b.grad) {
          const newGrad = await add(b.grad, dB)
          b.grad.dispose()
          dB.dispose()
          b.grad = newGrad
        } else {
          b.grad = dB
        }
      }
    })
  }
  return result
}

export async function div(a: Tensor, b: Tensor): Promise<Tensor> {
  return elementwiseBinary(a, b, 3)
}

// --- Elementwise Unary ---
async function elementwiseUnary(input: Tensor, opCode: number, scalar = 0): Promise<Tensor> {
  const device = await getGPUDevice()
  const result = await Tensor.empty(input.shape)
  const pipeline = await getPipeline(device, elementwiseUnaryShader, 'main')
  const uniform = createUniformBuffer(
    device,
    packMixed([
      { type: 'u32', value: input.size },
      { type: 'u32', value: opCode },
      { type: 'f32', value: scalar },
    ]),
  )
  dispatch(device, {
    pipeline,
    buffers: [input.buffer, result.buffer],
    uniformBuffer: uniform,
    workgroups: [Math.ceil(input.size / 256)],
  })
  await device.queue.onSubmittedWorkDone()
  uniform.destroy()
  return result
}

export async function relu(x: Tensor): Promise<Tensor> {
  return elementwiseUnary(x, 0)
}

export async function gelu(x: Tensor): Promise<Tensor> {
  const result = await elementwiseUnary(x, 1)
  if (isGradEnabled() && x.requiresGrad) {
    recordOp(result, [x], async (grad) => {
      const dX = await geluBackward(x, grad)
      if (x.grad) {
        const newGrad = await add(x.grad, dX)
        x.grad.dispose()
        dX.dispose()
        x.grad = newGrad
      } else {
        x.grad = dX
      }
    })
  }
  return result
}

export async function tanh_(x: Tensor): Promise<Tensor> {
  return elementwiseUnary(x, 2)
}

export async function exp_(x: Tensor): Promise<Tensor> {
  return elementwiseUnary(x, 3)
}

export async function neg(x: Tensor): Promise<Tensor> {
  return elementwiseUnary(x, 4)
}

export async function scalarMul(x: Tensor, s: number): Promise<Tensor> {
  return elementwiseUnary(x, 5, s)
}

// --- Copy ---
export async function copyTensor(src: Tensor): Promise<Tensor> {
  const device = await getGPUDevice()
  const dst = await Tensor.empty(src.shape)
  const encoder = device.createCommandEncoder()
  encoder.copyBufferToBuffer(src.buffer, 0, dst.buffer, 0, src.size * 4)
  device.queue.submit([encoder.finish()])
  await device.queue.onSubmittedWorkDone()
  return dst
}

// --- Matmul ---
export async function matmul(a: Tensor, b: Tensor): Promise<Tensor> {
  // a: [M, K], b: [K, N] -> result: [M, N]
  if (a.shape.length !== 2 || b.shape.length !== 2) {
    throw new Error('matmul expects 2D tensors')
  }
  const M = a.shape[0], K = a.shape[1], N = b.shape[1]
  if (K !== b.shape[0]) {
    throw new Error(`matmul shape mismatch: [${a.shape}] @ [${b.shape}]`)
  }
  const device = await getGPUDevice()
  const result = await Tensor.empty([M, N])
  const pipeline = await getPipeline(device, matmulShader, 'main')
  const uniform = createUniformBuffer(device, packU32(M, N, K))
  dispatch(device, {
    pipeline,
    buffers: [a.buffer, b.buffer, result.buffer],
    uniformBuffer: uniform,
    workgroups: [Math.ceil(N / 16), Math.ceil(M / 16)],
  })
  await device.queue.onSubmittedWorkDone()
  uniform.destroy()

  if (isGradEnabled() && (a.requiresGrad || b.requiresGrad)) {
    recordOp(result, [a, b], async (grad) => {
      // dA = grad @ B^T, dB = A^T @ grad
      if (a.requiresGrad) {
        const bT = await transpose(b)
        const dA = await matmul(grad, bT)
        bT.dispose()
        if (a.grad) {
          const newGrad = await add(a.grad, dA)
          a.grad.dispose()
          dA.dispose()
          a.grad = newGrad
        } else {
          a.grad = dA
        }
      }
      if (b.requiresGrad) {
        const aT = await transpose(a)
        const dB = await matmul(aT, grad)
        aT.dispose()
        if (b.grad) {
          const newGrad = await add(b.grad, dB)
          b.grad.dispose()
          dB.dispose()
          b.grad = newGrad
        } else {
          b.grad = dB
        }
      }
    })
  }
  return result
}

// --- Softmax ---
export async function softmax(input: Tensor): Promise<Tensor> {
  // input: [rows, cols], softmax along last dim
  if (input.shape.length !== 2) throw new Error('softmax expects 2D tensor')
  const rows = input.shape[0], cols = input.shape[1]
  const device = await getGPUDevice()
  const result = await Tensor.empty(input.shape)
  const pipeline = await getPipeline(device, softmaxShader, 'main')
  const uniform = createUniformBuffer(device, packU32(rows, cols))
  dispatch(device, {
    pipeline,
    buffers: [input.buffer, result.buffer],
    uniformBuffer: uniform,
    workgroups: [rows],
  })
  await device.queue.onSubmittedWorkDone()
  uniform.destroy()
  return result
}

// --- LayerNorm ---
export async function layernorm(
  input: Tensor,
  gamma: Tensor,
  beta: Tensor,
  eps = 1e-5,
): Promise<Tensor> {
  const rows = input.size / input.shape[input.shape.length - 1]
  const cols = input.shape[input.shape.length - 1]
  const device = await getGPUDevice()
  const result = await Tensor.empty(input.shape)
  const pipeline = await getPipeline(device, layernormShader, 'main')
  const uniform = createUniformBuffer(
    device,
    packMixed([
      { type: 'u32', value: rows },
      { type: 'u32', value: cols },
      { type: 'f32', value: eps },
    ]),
  )
  dispatch(device, {
    pipeline,
    buffers: [input.buffer, gamma.buffer, beta.buffer, result.buffer],
    uniformBuffer: uniform,
    workgroups: [rows],
  })
  await device.queue.onSubmittedWorkDone()
  uniform.destroy()

  if (isGradEnabled() && (input.requiresGrad || gamma.requiresGrad || beta.requiresGrad)) {
    recordOp(result, [input, gamma, beta], async (grad) => {
      await layernormBackward(input, gamma, beta, grad, eps)
    })
  }
  return result
}

// --- Embedding ---
export async function embedding(table: Tensor, indices: Tensor): Promise<Tensor> {
  // table: [V, D], indices: [N] -> result: [N, D]
  const D = table.shape[1]
  const N = indices.size
  const device = await getGPUDevice()
  const result = await Tensor.empty([N, D])
  const pipeline = await getPipeline(device, embeddingShader, 'main')
  const uniform = createUniformBuffer(device, packU32(N, D))
  dispatch(device, {
    pipeline,
    buffers: [table.buffer, indices.buffer, result.buffer],
    uniformBuffer: uniform,
    workgroups: [Math.ceil((N * D) / 256)],
  })
  await device.queue.onSubmittedWorkDone()
  uniform.destroy()

  if (isGradEnabled() && table.requiresGrad) {
    recordOp(result, [table, indices], async (grad) => {
      // CPU fallback scatter-add for embedding backward
      await embeddingBackward(table, indices, grad)
    })
  }
  return result
}

// --- Causal Mask ---
export async function causalMask(input: Tensor): Promise<Tensor> {
  // input: [rows, cols] - typically attention scores [T, T]
  if (input.shape.length !== 2) throw new Error('causalMask expects 2D tensor')
  const rows = input.shape[0], cols = input.shape[1]
  const device = await getGPUDevice()
  const result = await Tensor.empty(input.shape)
  const pipeline = await getPipeline(device, causalMaskShader, 'main')
  const uniform = createUniformBuffer(device, packU32(rows, cols))
  dispatch(device, {
    pipeline,
    buffers: [input.buffer, result.buffer],
    uniformBuffer: uniform,
    workgroups: [Math.ceil((rows * cols) / 256)],
  })
  await device.queue.onSubmittedWorkDone()
  uniform.destroy()
  return result
}

// --- Cross-entropy ---
export async function crossEntropy(logits: Tensor, targets: Tensor): Promise<Tensor> {
  // logits: [N, C], targets: [N] (u32)
  const N = logits.shape[0], C = logits.shape[1]
  const device = await getGPUDevice()
  const losses = await Tensor.empty([N])
  const pipeline = await getPipeline(device, crossentropyShader, 'main')
  const uniform = createUniformBuffer(device, packU32(N, C))
  dispatch(device, {
    pipeline,
    buffers: [logits.buffer, targets.buffer, losses.buffer],
    uniformBuffer: uniform,
    workgroups: [Math.ceil(N / 256)],
  })
  await device.queue.onSubmittedWorkDone()

  // Reduce to mean
  const meanLoss = await reduceMean(losses, N)
  uniform.destroy()

  if (isGradEnabled() && logits.requiresGrad) {
    recordOp(meanLoss, [logits, targets], async (grad) => {
      await crossEntropyBackward(logits, targets, grad)
    })
  }

  losses.dispose()
  return meanLoss
}

// --- Reduce mean ---
async function reduceMean(input: Tensor, count: number): Promise<Tensor> {
  const device = await getGPUDevice()
  const sumResult = await Tensor.empty([1])
  const pipeline = await getPipeline(device, reduceShader, 'main')
  const uniform = createUniformBuffer(device, packU32(input.size))
  dispatch(device, {
    pipeline,
    buffers: [input.buffer, sumResult.buffer],
    uniformBuffer: uniform,
    workgroups: [1],
  })
  await device.queue.onSubmittedWorkDone()
  uniform.destroy()

  // Divide by count
  const divisor = await Tensor.create(new Float32Array([count]), [1])
  const result = await elementwiseBinary(sumResult, divisor, 3) // div
  sumResult.dispose()
  divisor.dispose()
  return result
}

// --- Transpose ---
export async function transpose(input: Tensor): Promise<Tensor> {
  if (input.shape.length !== 2) throw new Error('transpose expects 2D tensor')
  const rows = input.shape[0], cols = input.shape[1]
  const device = await getGPUDevice()
  const result = await Tensor.empty([cols, rows])
  const pipeline = await getPipeline(device, transposeShader, 'main')
  const uniform = createUniformBuffer(device, packU32(rows, cols))
  dispatch(device, {
    pipeline,
    buffers: [input.buffer, result.buffer],
    uniformBuffer: uniform,
    workgroups: [Math.ceil((rows * cols) / 256)],
  })
  await device.queue.onSubmittedWorkDone()
  uniform.destroy()
  return result
}

// --- Backward ops (forward declarations, implemented in backward.ts) ---
// These are lazily imported to avoid circular deps

async function geluBackward(x: Tensor, grad: Tensor): Promise<Tensor> {
  const { geluBackwardOp } = await import('./backward')
  return geluBackwardOp(x, grad)
}

async function layernormBackward(
  input: Tensor,
  gamma: Tensor,
  beta: Tensor,
  grad: Tensor,
  eps: number,
): Promise<void> {
  const { layernormBackwardFull } = await import('./backward')
  return layernormBackwardFull(input, gamma, beta, grad, eps)
}

async function embeddingBackward(
  table: Tensor,
  indices: Tensor,
  grad: Tensor,
): Promise<void> {
  const { embeddingBackwardOp } = await import('./backward')
  return embeddingBackwardOp(table, indices, grad)
}

async function crossEntropyBackward(
  logits: Tensor,
  targets: Tensor,
  grad: Tensor,
): Promise<void> {
  const { crossEntropyBackwardOp } = await import('./backward')
  return crossEntropyBackwardOp(logits, targets, grad)
}

// --- Multi-head Attention (composed) ---
export async function multiHeadAttention(
  Q: Tensor, K: Tensor, V: Tensor,
  nHeads: number, seqLen: number, dModel: number,
): Promise<Tensor> {
  const headDim = dModel / nHeads
  const scale = 1 / Math.sqrt(headDim)
  const B = Q.shape[0] / seqLen // batch size (Q is [B*T, dModel])

  // We'll process each batch*head by slicing and doing 2D matmuls
  // For Phase 1: reshape to [B*nHeads, T, headDim] by reading back and re-uploading
  // This is slow but correct; Phase 2 will use batched matmul kernels

  const qArr = await Q.toArray()
  const kArr = await K.toArray()
  const vArr = await V.toArray()

  const outArr = new Float32Array(B * seqLen * dModel)

  for (let b = 0; b < B; b++) {
    for (let h = 0; h < nHeads; h++) {
      // Extract head slices: [T, headDim]
      const qHead = new Float32Array(seqLen * headDim)
      const kHead = new Float32Array(seqLen * headDim)
      const vHead = new Float32Array(seqLen * headDim)

      for (let t = 0; t < seqLen; t++) {
        const srcOffset = (b * seqLen + t) * dModel + h * headDim
        const dstOffset = t * headDim
        qHead.set(qArr.subarray(srcOffset, srcOffset + headDim), dstOffset)
        kHead.set(kArr.subarray(srcOffset, srcOffset + headDim), dstOffset)
        vHead.set(vArr.subarray(srcOffset, srcOffset + headDim), dstOffset)
      }

      // Q @ K^T -> [T, T]
      const qTensor = await Tensor.create(qHead, [seqLen, headDim])
      const kTensor = await Tensor.create(kHead, [seqLen, headDim])
      const vTensor = await Tensor.create(vHead, [seqLen, headDim])

      const kT = await transpose(kTensor)
      let scores = await matmul(qTensor, kT)
      kT.dispose()

      // Scale
      const scaleTensor = await Tensor.create(
        new Float32Array(scores.size).fill(scale),
        scores.shape,
      )
      let scaledScores = await elementwiseBinary(scores, scaleTensor, 2)
      scores.dispose()
      scaleTensor.dispose()

      // Causal mask
      const masked = await causalMask(scaledScores)
      scaledScores.dispose()

      // Softmax
      const attnWeights = await softmax(masked)
      masked.dispose()

      // Attn @ V -> [T, headDim]
      const headOut = await matmul(attnWeights, vTensor)
      attnWeights.dispose()

      // Write back to output
      const headOutArr = await headOut.toArray()
      for (let t = 0; t < seqLen; t++) {
        const dstOffset = (b * seqLen + t) * dModel + h * headDim
        const srcOffset = t * headDim
        outArr.set(headOutArr.subarray(srcOffset, srcOffset + headDim), dstOffset)
      }

      qTensor.dispose()
      kTensor.dispose()
      vTensor.dispose()
      headOut.dispose()
    }
  }

  return Tensor.create(outArr, Q.shape)
}
