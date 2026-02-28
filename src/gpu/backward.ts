import { getGPUDevice } from './device'
import { Tensor } from './tensor'
import { add } from './ops'

import geluBackwardShader from './ops/gelu_backward.wgsl'
import crossentropyBackwardShader from './ops/crossentropy_backward.wgsl'
import layernormBackwardShader from './ops/layernorm_backward.wgsl'

// --- Pipeline cache (shared pattern) ---
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

function createUniformBuffer(device: GPUDevice, data: ArrayBuffer): GPUBuffer {
  const buf = device.createBuffer({
    size: Math.max(data.byteLength, 16),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  })
  new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data))
  buf.unmap()
  return buf
}

function packU32(...values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length * 4)
  new Uint32Array(buf).set(values)
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

function dispatch(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  buffers: GPUBuffer[],
  uniformBuffer: GPUBuffer | null,
  workgroups: [number, number?, number?],
): void {
  const entries: GPUBindGroupEntry[] = buffers.map((b, i) => ({
    binding: i,
    resource: { buffer: b },
  }))
  if (uniformBuffer) {
    entries.push({
      binding: buffers.length,
      resource: { buffer: uniformBuffer },
    })
  }
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries,
  })
  const encoder = device.createCommandEncoder()
  const pass = encoder.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(workgroups[0], workgroups[1] ?? 1, workgroups[2] ?? 1)
  pass.end()
  device.queue.submit([encoder.finish()])
}

// --- GELU Backward ---
export async function geluBackwardOp(x: Tensor, grad: Tensor): Promise<Tensor> {
  const device = await getGPUDevice()
  const result = await Tensor.empty(x.shape)
  const pipeline = await getPipeline(device, geluBackwardShader, 'main')
  const uniform = createUniformBuffer(device, packU32(x.size))
  dispatch(device, pipeline, [x.buffer, grad.buffer, result.buffer], uniform, [
    Math.ceil(x.size / 256),
  ])
  await device.queue.onSubmittedWorkDone()
  uniform.destroy()
  return result
}

// --- Cross-entropy Backward ---
export async function crossEntropyBackwardOp(
  logits: Tensor,
  targets: Tensor,
  grad: Tensor,
): Promise<void> {
  const N = logits.shape[0], C = logits.shape[1]
  const device = await getGPUDevice()
  const gradLogits = await Tensor.empty(logits.shape)
  const pipeline = await getPipeline(device, crossentropyBackwardShader, 'main')
  const uniform = createUniformBuffer(device, packU32(N, C))
  dispatch(
    device,
    pipeline,
    [logits.buffer, targets.buffer, grad.buffer, gradLogits.buffer],
    uniform,
    [Math.ceil(N / 256)],
  )
  await device.queue.onSubmittedWorkDone()
  uniform.destroy()

  if (logits.grad) {
    const newGrad = await add(logits.grad, gradLogits)
    logits.grad.dispose()
    gradLogits.dispose()
    logits.grad = newGrad
  } else {
    logits.grad = gradLogits
  }
}

// --- LayerNorm Backward ---
export async function layernormBackwardOp(
  input: Tensor,
  gamma: Tensor,
  grad: Tensor,
  eps: number,
): Promise<void> {
  const cols = input.shape[input.shape.length - 1]
  const rows = input.size / cols
  const device = await getGPUDevice()

  const gradInput = await Tensor.empty(input.shape)
  // For grad_gamma and grad_beta, we need per-column accumulation across rows
  // The kernel writes per-row contributions; we need a separate accumulation
  // For Phase 1: we'll do grad_gamma/grad_beta on CPU
  const gradGamma = await Tensor.zeros([cols])
  const gradBeta = await Tensor.zeros([cols])

  const pipeline = await getPipeline(device, layernormBackwardShader, 'main')
  const uniform = createUniformBuffer(
    device,
    packMixed([
      { type: 'u32', value: rows },
      { type: 'u32', value: cols },
      { type: 'f32', value: eps },
    ]),
  )
  dispatch(
    device,
    pipeline,
    [
      input.buffer,
      gamma.buffer,
      grad.buffer,
      gradInput.buffer,
      gradGamma.buffer,
      gradBeta.buffer,
    ],
    uniform,
    [rows],
  )
  await device.queue.onSubmittedWorkDone()
  uniform.destroy()

  // CPU accumulation for grad_gamma and grad_beta
  const inputArr = await input.toArray()
  const gradArr = await grad.toArray()
  const ggArr = new Float32Array(cols)
  const gbArr = new Float32Array(cols)

  for (let r = 0; r < rows; r++) {
    const offset = r * cols
    // Recompute mean and rstd for this row
    let mean = 0
    for (let c = 0; c < cols; c++) mean += inputArr[offset + c]
    mean /= cols
    let variance = 0
    for (let c = 0; c < cols; c++) {
      const diff = inputArr[offset + c] - mean
      variance += diff * diff
    }
    variance /= cols
    const rstd = 1 / Math.sqrt(variance + eps)

    for (let c = 0; c < cols; c++) {
      const normalized = (inputArr[offset + c] - mean) * rstd
      ggArr[c] += gradArr[offset + c] * normalized
      gbArr[c] += gradArr[offset + c]
    }
  }

  // Upload accumulated grad_gamma/grad_beta
  const ggTensor = await Tensor.create(ggArr, [cols])
  const gbTensor = await Tensor.create(gbArr, [cols])

  // Accumulate into param grads
  if (input.requiresGrad) {
    if (input.grad) {
      const newGrad = await add(input.grad, gradInput)
      input.grad.dispose()
      gradInput.dispose()
      input.grad = newGrad
    } else {
      input.grad = gradInput
    }
  } else {
    gradInput.dispose()
  }

  if (gamma.requiresGrad) {
    if (gamma.grad) {
      const newGrad = await add(gamma.grad, ggTensor)
      gamma.grad.dispose()
      ggTensor.dispose()
      gamma.grad = newGrad
    } else {
      gamma.grad = ggTensor
    }
  } else {
    ggTensor.dispose()
  }

  // beta (the third parent passed to layernorm)
  // We need to find the beta tensor - it was the 3rd arg to layernorm
  // In our autograd recording, parents[2] is beta
  // Since this function receives gamma directly, we need to handle beta grad
  // through the original recording. Let's use a workaround:
  // The caller in ops.ts recorded [input, gamma, beta] as parents
  // We can check gammaArr length to find beta... actually we receive it from the caller

  // For simplicity, we'll set gradBeta on a global that the caller reads
  // Actually, let's just not - the ops.ts layernormBackward passes input, gamma
  // We need to also pass beta. Let's use a different approach:
  // Store grad_beta in the gbTensor and let it be picked up

  // Actually the simplest fix: we'll export a function that also takes beta
  gbTensor.dispose()
  gradGamma.dispose()
  gradBeta.dispose()
}

// Overload that takes beta
export async function layernormBackwardFull(
  input: Tensor,
  gamma: Tensor,
  beta: Tensor,
  grad: Tensor,
  eps: number,
): Promise<void> {
  const cols = input.shape[input.shape.length - 1]
  const rows = input.size / cols

  // CPU-based backward for Phase 1 (simpler and correct)
  const inputArr = await input.toArray()
  const gradArr = await grad.toArray()
  const gammaArr = await gamma.toArray()

  const gradInputArr = new Float32Array(input.size)
  const ggArr = new Float32Array(cols)
  const gbArr = new Float32Array(cols)

  for (let r = 0; r < rows; r++) {
    const offset = r * cols
    let mean = 0
    for (let c = 0; c < cols; c++) mean += inputArr[offset + c]
    mean /= cols
    let variance = 0
    for (let c = 0; c < cols; c++) {
      const diff = inputArr[offset + c] - mean
      variance += diff * diff
    }
    variance /= cols
    const rstd = 1 / Math.sqrt(variance + eps)

    // Accumulate grad_gamma, grad_beta
    let ds = 0, db = 0
    for (let c = 0; c < cols; c++) {
      const g = gradArr[offset + c] * gammaArr[c]
      const xhat = (inputArr[offset + c] - mean) * rstd
      ggArr[c] += gradArr[offset + c] * xhat
      gbArr[c] += gradArr[offset + c]
      ds += g * (inputArr[offset + c] - mean)
      db += g
    }

    // grad_input
    for (let c = 0; c < cols; c++) {
      const g = gradArr[offset + c] * gammaArr[c]
      const xhat = (inputArr[offset + c] - mean) * rstd
      gradInputArr[offset + c] = rstd * (g - (db + xhat * ds * rstd) / cols)
    }
  }

  const gradInput = await Tensor.create(gradInputArr, input.shape)
  const ggTensor = await Tensor.create(ggArr, [cols])
  const gbTensor = await Tensor.create(gbArr, [cols])

  if (input.requiresGrad) {
    if (input.grad) {
      const n = await add(input.grad, gradInput)
      input.grad.dispose()
      gradInput.dispose()
      input.grad = n
    } else {
      input.grad = gradInput
    }
  } else {
    gradInput.dispose()
  }

  if (gamma.requiresGrad) {
    if (gamma.grad) {
      const n = await add(gamma.grad, ggTensor)
      gamma.grad.dispose()
      ggTensor.dispose()
      gamma.grad = n
    } else {
      gamma.grad = ggTensor
    }
  } else {
    ggTensor.dispose()
  }

  if (beta.requiresGrad) {
    if (beta.grad) {
      const n = await add(beta.grad, gbTensor)
      beta.grad.dispose()
      gbTensor.dispose()
      beta.grad = n
    } else {
      beta.grad = gbTensor
    }
  } else {
    gbTensor.dispose()
  }
}

// --- Embedding Backward (CPU scatter-add) ---
export async function embeddingBackwardOp(
  table: Tensor,
  indices: Tensor,
  grad: Tensor,
): Promise<void> {
  const V = table.shape[0], D = table.shape[1]
  const N = indices.size

  // Read indices and grad to CPU
  const device = await getGPUDevice()
  // Read indices as u32
  const idxStaging = device.createBuffer({
    size: N * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  })
  const encoder = device.createCommandEncoder()
  encoder.copyBufferToBuffer(indices.buffer, 0, idxStaging, 0, N * 4)
  device.queue.submit([encoder.finish()])
  await idxStaging.mapAsync(GPUMapMode.READ)
  const idxArr = new Uint32Array(idxStaging.getMappedRange()).slice()
  idxStaging.unmap()
  idxStaging.destroy()

  const gradArr = await grad.toArray()

  // Scatter-add
  const gradTable = new Float32Array(V * D)
  for (let i = 0; i < N; i++) {
    const vocabIdx = idxArr[i]
    for (let d = 0; d < D; d++) {
      gradTable[vocabIdx * D + d] += gradArr[i * D + d]
    }
  }

  const gradTensor = await Tensor.create(gradTable, [V, D])
  if (table.grad) {
    const newGrad = await add(table.grad, gradTensor)
    table.grad.dispose()
    gradTensor.dispose()
    table.grad = newGrad
  } else {
    table.grad = gradTensor
  }
}
