import { getGPUDevice } from './device'
import { Tensor } from './tensor'
import { recordOp, isGradEnabled } from './autograd'

import elementwiseBinaryShader from './ops/elementwise_binary.wgsl'
import elementwiseUnaryShader from './ops/elementwise_unary.wgsl'
import matmulShader from './ops/matmul.wgsl'
import batchedMatmulShader from './ops/batched_matmul.wgsl'
import batchedTransposeShader from './ops/batched_transpose.wgsl'
import reshapeHeadsShader from './ops/reshape_heads.wgsl'
import softmaxShader from './ops/softmax.wgsl'
import softmaxBackwardShader from './ops/softmax_backward.wgsl'
import layernormShader from './ops/layernorm.wgsl'
import embeddingShader from './ops/embedding.wgsl'
import causalMaskShader from './ops/causal_mask.wgsl'
import causalMaskBackwardShader from './ops/causal_mask_backward.wgsl'
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
  const result = await elementwiseUnary(x, 5, s)
  if (isGradEnabled() && x.requiresGrad) {
    recordOp(result, [x], async (grad) => {
      const dX = await scalarMul(grad, s)
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
  // input: [rows, cols] or [B, rows, cols], softmax along last dim
  if (input.shape.length !== 2 && input.shape.length !== 3) {
    throw new Error('softmax expects 2D or 3D tensor')
  }
  const cols = input.shape[input.shape.length - 1]
  const rows = input.size / cols  // flatten leading dims
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

  if (isGradEnabled() && input.requiresGrad) {
    recordOp(result, [input], async (grad) => {
      await softmaxBackward(result, grad, input)
    })
  }
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
  // input: [T, T] or [B, T, T] - attention scores
  if (input.shape.length !== 2 && input.shape.length !== 3) {
    throw new Error('causalMask expects 2D or 3D tensor')
  }
  const T = input.shape[input.shape.length - 1]
  const totalSlices = input.size / (T * T)
  const device = await getGPUDevice()
  const result = await Tensor.empty(input.shape)
  const pipeline = await getPipeline(device, causalMaskShader, 'main')
  const uniform = createUniformBuffer(device, packU32(totalSlices, T))
  dispatch(device, {
    pipeline,
    buffers: [input.buffer, result.buffer],
    uniformBuffer: uniform,
    workgroups: [Math.ceil(input.size / 256)],
  })
  await device.queue.onSubmittedWorkDone()
  uniform.destroy()

  if (isGradEnabled() && input.requiresGrad) {
    recordOp(result, [input], async (grad) => {
      await causalMaskBackward(grad, T, totalSlices, input)
    })
  }
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

  if (isGradEnabled() && input.requiresGrad) {
    recordOp(result, [input], async (grad) => {
      // Backward of transpose is transpose
      const dInput = await transpose(grad)
      if (input.grad) {
        const newGrad = await add(input.grad, dInput)
        input.grad.dispose()
        dInput.dispose()
        input.grad = newGrad
      } else {
        input.grad = dInput
      }
    })
  }
  return result
}

// --- Batched Matmul ---
export async function batchedMatmul(a: Tensor, b: Tensor): Promise<Tensor> {
  // a: [B, M, K], b: [B, K, N] -> result: [B, M, N]
  if (a.shape.length !== 3 || b.shape.length !== 3) {
    throw new Error('batchedMatmul expects 3D tensors')
  }
  const B = a.shape[0], M = a.shape[1], K = a.shape[2], N = b.shape[2]
  if (a.shape[0] !== b.shape[0] || K !== b.shape[1]) {
    throw new Error(`batchedMatmul shape mismatch: [${a.shape}] @ [${b.shape}]`)
  }
  const device = await getGPUDevice()
  const result = await Tensor.empty([B, M, N])
  const pipeline = await getPipeline(device, batchedMatmulShader, 'main')
  const uniform = createUniformBuffer(device, packU32(B, M, N, K))
  dispatch(device, {
    pipeline,
    buffers: [a.buffer, b.buffer, result.buffer],
    uniformBuffer: uniform,
    workgroups: [Math.ceil(N / 16), Math.ceil(M / 16), B],
  })
  await device.queue.onSubmittedWorkDone()
  uniform.destroy()

  if (isGradEnabled() && (a.requiresGrad || b.requiresGrad)) {
    recordOp(result, [a, b], async (grad) => {
      // dA = grad @ batchedTranspose(B), dB = batchedTranspose(A) @ grad
      if (a.requiresGrad) {
        const bT = await batchedTranspose(b)
        const dA = await batchedMatmul(grad, bT)
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
        const aT = await batchedTranspose(a)
        const dB = await batchedMatmul(aT, grad)
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

// --- Batched Transpose ---
export async function batchedTranspose(input: Tensor): Promise<Tensor> {
  // input: [B, R, C] -> result: [B, C, R]
  if (input.shape.length !== 3) throw new Error('batchedTranspose expects 3D tensor')
  const B = input.shape[0], R = input.shape[1], C = input.shape[2]
  const device = await getGPUDevice()
  const result = await Tensor.empty([B, C, R])
  const pipeline = await getPipeline(device, batchedTransposeShader, 'main')
  const uniform = createUniformBuffer(device, packU32(B, R, C))
  dispatch(device, {
    pipeline,
    buffers: [input.buffer, result.buffer],
    uniformBuffer: uniform,
    workgroups: [Math.ceil((R * C) / 256), B],
  })
  await device.queue.onSubmittedWorkDone()
  uniform.destroy()

  if (isGradEnabled() && input.requiresGrad) {
    recordOp(result, [input], async (grad) => {
      // Backward of transpose is transpose
      const dInput = await batchedTranspose(grad)
      if (input.grad) {
        const newGrad = await add(input.grad, dInput)
        input.grad.dispose()
        dInput.dispose()
        input.grad = newGrad
      } else {
        input.grad = dInput
      }
    })
  }
  return result
}

// --- Reshape Heads ---
export async function reshapeHeads(
  input: Tensor, B: number, T: number, nHeads: number,
): Promise<Tensor> {
  // input: [B*T, D] -> result: [B*nHeads, T, headDim]
  if (input.shape.length !== 2) throw new Error('reshapeHeads expects 2D tensor')
  const D = input.shape[1]
  const headDim = D / nHeads
  const device = await getGPUDevice()
  const result = await Tensor.empty([B * nHeads, T, headDim])
  const pipeline = await getPipeline(device, reshapeHeadsShader, 'main')
  const uniform = createUniformBuffer(device, packU32(B, T, nHeads, headDim, 0)) // direction=0 (split)
  const totalElements = B * nHeads * T * headDim
  dispatch(device, {
    pipeline,
    buffers: [input.buffer, result.buffer],
    uniformBuffer: uniform,
    workgroups: [Math.ceil(totalElements / 256)],
  })
  await device.queue.onSubmittedWorkDone()
  uniform.destroy()

  if (isGradEnabled() && input.requiresGrad) {
    recordOp(result, [input], async (grad) => {
      const dInput = await mergeHeads(grad, B, T, nHeads)
      if (input.grad) {
        const newGrad = await add(input.grad, dInput)
        input.grad.dispose()
        dInput.dispose()
        input.grad = newGrad
      } else {
        input.grad = dInput
      }
    })
  }
  return result
}

// --- Merge Heads ---
export async function mergeHeads(
  input: Tensor, B: number, T: number, nHeads: number,
): Promise<Tensor> {
  // input: [B*nHeads, T, headDim] -> result: [B*T, D]
  if (input.shape.length !== 3) throw new Error('mergeHeads expects 3D tensor')
  const headDim = input.shape[2]
  const D = nHeads * headDim
  const device = await getGPUDevice()
  const result = await Tensor.empty([B * T, D])
  const pipeline = await getPipeline(device, reshapeHeadsShader, 'main')
  const uniform = createUniformBuffer(device, packU32(B, T, nHeads, headDim, 1)) // direction=1 (merge)
  const totalElements = B * nHeads * T * headDim
  dispatch(device, {
    pipeline,
    buffers: [input.buffer, result.buffer],
    uniformBuffer: uniform,
    workgroups: [Math.ceil(totalElements / 256)],
  })
  await device.queue.onSubmittedWorkDone()
  uniform.destroy()

  if (isGradEnabled() && input.requiresGrad) {
    recordOp(result, [input], async (grad) => {
      const dInput = await reshapeHeads(grad, B, T, nHeads)
      if (input.grad) {
        const newGrad = await add(input.grad, dInput)
        input.grad.dispose()
        dInput.dispose()
        input.grad = newGrad
      } else {
        input.grad = dInput
      }
    })
  }
  return result
}

// --- Softmax Backward ---
async function softmaxBackward(
  softmaxOut: Tensor, grad: Tensor, input: Tensor,
): Promise<void> {
  const cols = softmaxOut.shape[softmaxOut.shape.length - 1]
  const rows = softmaxOut.size / cols
  const device = await getGPUDevice()
  const result = await Tensor.empty(softmaxOut.shape)
  const pipeline = await getPipeline(device, softmaxBackwardShader, 'main')
  const uniform = createUniformBuffer(device, packU32(rows, cols))
  dispatch(device, {
    pipeline,
    buffers: [softmaxOut.buffer, grad.buffer, result.buffer],
    uniformBuffer: uniform,
    workgroups: [rows],
  })
  await device.queue.onSubmittedWorkDone()
  uniform.destroy()

  if (input.requiresGrad) {
    if (input.grad) {
      const newGrad = await add(input.grad, result)
      input.grad.dispose()
      result.dispose()
      input.grad = newGrad
    } else {
      input.grad = result
    }
  } else {
    result.dispose()
  }
}

// --- Causal Mask Backward ---
async function causalMaskBackward(
  grad: Tensor, T: number, totalSlices: number, input: Tensor,
): Promise<void> {
  const device = await getGPUDevice()
  const result = await Tensor.empty(grad.shape)
  const pipeline = await getPipeline(device, causalMaskBackwardShader, 'main')
  const uniform = createUniformBuffer(device, packU32(totalSlices, T))
  dispatch(device, {
    pipeline,
    buffers: [grad.buffer, result.buffer],
    uniformBuffer: uniform,
    workgroups: [Math.ceil(grad.size / 256)],
  })
  await device.queue.onSubmittedWorkDone()
  uniform.destroy()

  if (input.requiresGrad) {
    if (input.grad) {
      const newGrad = await add(input.grad, result)
      input.grad.dispose()
      result.dispose()
      input.grad = newGrad
    } else {
      input.grad = result
    }
  } else {
    result.dispose()
  }
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

// --- Multi-head Attention (GPU-native with full autograd) ---
export async function multiHeadAttention(
  Q: Tensor, K: Tensor, V: Tensor,
  nHeads: number, seqLen: number, dModel: number,
): Promise<Tensor> {
  const headDim = dModel / nHeads
  const scale = 1 / Math.sqrt(headDim)
  const B = Q.shape[0] / seqLen // batch size (Q is [B*T, dModel])
  const grad = isGradEnabled()

  // 1. Reshape Q, K, V: [B*T, D] -> [B*nHeads, T, headDim]
  const Qh = await reshapeHeads(Q, B, seqLen, nHeads)
  const Kh = await reshapeHeads(K, B, seqLen, nHeads)
  const Vh = await reshapeHeads(V, B, seqLen, nHeads)

  // 2. KhT = batchedTranspose(Kh): [B*nH, headDim, T]
  const KhT = await batchedTranspose(Kh)

  // 3. scores = Qh @ KhT: [B*nH, T, T]
  const scores = await batchedMatmul(Qh, KhT)
  // Don't dispose KhT/Qh yet - batchedMatmul backward needs them
  if (!grad) { KhT.dispose(); Qh.dispose() }

  // 4. Scale scores
  const scaledScores = await scalarMul(scores, scale)
  if (!grad) scores.dispose()

  // 5. Causal mask
  const masked = await causalMask(scaledScores)
  if (!grad) scaledScores.dispose()

  // 6. Softmax
  const attnWeights = await softmax(masked)
  if (!grad) masked.dispose()

  // 7. attnOut = attnWeights @ Vh: [B*nH, T, headDim]
  const attnOut = await batchedMatmul(attnWeights, Vh)
  // batchedMatmul backward needs attnWeights and Vh
  if (!grad) { Kh.dispose(); Vh.dispose(); attnWeights.dispose() }

  // 8. Merge heads: [B*nH, T, headDim] -> [B*T, D]
  const output = await mergeHeads(attnOut, B, seqLen, nHeads)
  if (!grad) attnOut.dispose()

  return output
}
