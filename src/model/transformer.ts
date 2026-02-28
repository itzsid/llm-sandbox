import { Tensor } from '../gpu/tensor'
import { add, matmul, gelu, layernorm, embedding, multiHeadAttention } from '../gpu/ops'
import { isGradEnabled } from '../gpu/autograd'
import type { TransformerConfig } from './config'

export interface LayerParams {
  lnAttnGamma: Tensor
  lnAttnBeta: Tensor
  Wq: Tensor
  Wk: Tensor
  Wv: Tensor
  Wo: Tensor
  lnFFGamma: Tensor
  lnFFBeta: Tensor
  W1: Tensor  // FFN up projection [dModel, dFF]
  b1: Tensor  // FFN bias [dFF]
  W2: Tensor  // FFN down projection [dFF, dModel]
  b2: Tensor  // FFN bias [dModel]
}

export interface TransformerParams {
  tokenEmbed: Tensor   // [vocabSize, dModel]
  posEmbed: Tensor     // [blockSize, dModel]
  layers: LayerParams[]
  lnFinalGamma: Tensor // [dModel]
  lnFinalBeta: Tensor  // [dModel]
  lmHead: Tensor       // [dModel, vocabSize]
}

function xavierInit(shape: number[]): Float32Array {
  const size = shape.reduce((a, b) => a * b, 1)
  const fanIn = shape.length > 1 ? shape[shape.length - 2] : shape[0]
  const fanOut = shape[shape.length - 1]
  const std = Math.sqrt(2.0 / (fanIn + fanOut))
  const data = new Float32Array(size)
  for (let i = 0; i < size; i++) {
    // Box-Muller transform
    const u1 = Math.random()
    const u2 = Math.random()
    data[i] = std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
  }
  return data
}

export async function initTransformer(config: TransformerConfig): Promise<TransformerParams> {
  const { vocabSize, blockSize, nLayers, dModel, dFF } = config

  const tokenEmbed = await Tensor.create(
    xavierInit([vocabSize, dModel]),
    [vocabSize, dModel],
    { requiresGrad: true },
  )
  const posEmbed = await Tensor.create(
    xavierInit([blockSize, dModel]),
    [blockSize, dModel],
    { requiresGrad: true },
  )

  const layers: LayerParams[] = []
  for (let i = 0; i < nLayers; i++) {
    layers.push({
      lnAttnGamma: await Tensor.ones([dModel], { requiresGrad: true }),
      lnAttnBeta: await Tensor.zeros([dModel], { requiresGrad: true }),
      Wq: await Tensor.create(xavierInit([dModel, dModel]), [dModel, dModel], { requiresGrad: true }),
      Wk: await Tensor.create(xavierInit([dModel, dModel]), [dModel, dModel], { requiresGrad: true }),
      Wv: await Tensor.create(xavierInit([dModel, dModel]), [dModel, dModel], { requiresGrad: true }),
      Wo: await Tensor.create(xavierInit([dModel, dModel]), [dModel, dModel], { requiresGrad: true }),
      lnFFGamma: await Tensor.ones([dModel], { requiresGrad: true }),
      lnFFBeta: await Tensor.zeros([dModel], { requiresGrad: true }),
      W1: await Tensor.create(xavierInit([dModel, dFF]), [dModel, dFF], { requiresGrad: true }),
      b1: await Tensor.zeros([dFF], { requiresGrad: true }),
      W2: await Tensor.create(xavierInit([dFF, dModel]), [dFF, dModel], { requiresGrad: true }),
      b2: await Tensor.zeros([dModel], { requiresGrad: true }),
    })
  }

  const lnFinalGamma = await Tensor.ones([dModel], { requiresGrad: true })
  const lnFinalBeta = await Tensor.zeros([dModel], { requiresGrad: true })
  const lmHead = await Tensor.create(
    xavierInit([dModel, vocabSize]),
    [dModel, vocabSize],
    { requiresGrad: true },
  )

  return { tokenEmbed, posEmbed, layers, lnFinalGamma, lnFinalBeta, lmHead }
}

export async function transformerForward(
  params: TransformerParams,
  inputIds: Tensor,
  config: TransformerConfig,
): Promise<Tensor> {
  const B = inputIds.shape[0]
  const T = inputIds.shape[1]
  const { nHeads, dModel } = config
  // When grad is enabled (training), keep intermediates alive for backward.
  // When grad is disabled (generation), dispose eagerly to avoid GPU memory leaks.
  const training = isGradEnabled()

  // Flatten inputIds to [B*T] for embedding lookup
  // inputIds is [B, T] stored as contiguous u32
  const flatIds = inputIds // already flat in memory: [B*T]

  // Token embedding: [B*T, dModel]
  let x = await embedding(params.tokenEmbed, flatIds)

  // Position embedding: need to create position indices [B*T]
  // Positions: [0, 1, 2, ..., T-1] repeated B times
  const posIndices = new Uint32Array(B * T)
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      posIndices[b * T + t] = t
    }
  }
  const posIdsTensor = await Tensor.fromU32(posIndices, [B * T])
  const posEmbed = await embedding(params.posEmbed, posIdsTensor)
  if (!training) posIdsTensor.dispose()
  // When training: posIdsTensor kept alive — embedding backward needs indices.buffer

  // Add token + position embeddings
  const xPlusPos = await add(x, posEmbed)
  x.dispose()
  posEmbed.dispose()
  x = xPlusPos

  // Transformer layers
  for (const layer of params.layers) {
    const prevX = x // keep ref for conditional disposal

    // Pre-norm attention
    const normed1 = await layernorm(x, layer.lnAttnGamma, layer.lnAttnBeta)

    // Q, K, V projections: [B*T, dModel] @ [dModel, dModel] -> [B*T, dModel]
    const Q = await matmul(normed1, layer.Wq)
    const K = await matmul(normed1, layer.Wk)
    const V = await matmul(normed1, layer.Wv)
    if (!training) normed1.dispose()
    // When training: normed1 kept alive — matmul backward needs it for transpose(a)

    // Multi-head attention
    const attnOut = await multiHeadAttention(Q, K, V, nHeads, T, dModel)
    if (!training) { Q.dispose(); K.dispose(); V.dispose() }

    // Output projection
    const projected = await matmul(attnOut, layer.Wo)
    if (!training) attnOut.dispose()
    // When training: attnOut kept alive — matmul backward needs it

    // Residual connection
    const residual1 = await add(x, projected)
    if (!training) prevX.dispose()
    // When training: x kept alive — layernorm backward needs input.buffer
    projected.dispose()
    x = residual1

    const prevX2 = x // keep ref for conditional disposal

    // Pre-norm FFN
    const normed2 = await layernorm(x, layer.lnFFGamma, layer.lnFFBeta)

    // FFN: linear -> GELU -> linear
    let ffn = await matmul(normed2, layer.W1)
    if (!training) normed2.dispose()
    // When training: normed2 kept alive — matmul backward needs it

    // Add bias (broadcast: [B*T, dFF] + [dFF])
    const b1Tiled = await tileBias(layer.b1, B * T)
    const ffnBiased = await add(ffn, b1Tiled)
    ffn.dispose()
    b1Tiled.dispose()
    ffn = ffnBiased

    const activated = await gelu(ffn)
    if (!training) ffn.dispose()
    // When training: ffn kept alive — gelu backward needs input.buffer

    let ffnOut = await matmul(activated, layer.W2)
    if (!training) activated.dispose()
    // When training: activated kept alive — matmul backward needs it

    const b2Tiled = await tileBias(layer.b2, B * T)
    const ffnOutBiased = await add(ffnOut, b2Tiled)
    ffnOut.dispose()
    b2Tiled.dispose()
    ffnOut = ffnOutBiased

    // Residual connection
    const residual2 = await add(x, ffnOut)
    if (!training) prevX2.dispose()
    // When training: x kept alive — layernorm backward needs input.buffer
    ffnOut.dispose()
    x = residual2
  }

  // Final layer norm
  const prevXFinal = x
  const normedFinal = await layernorm(x, params.lnFinalGamma, params.lnFinalBeta)
  if (!training) prevXFinal.dispose()
  // When training: x kept alive — layernorm backward needs input.buffer

  // LM head: [B*T, dModel] @ [dModel, vocabSize] -> [B*T, vocabSize]
  const logits = await matmul(normedFinal, params.lmHead)
  if (!training) normedFinal.dispose()
  // When training: normedFinal kept alive — matmul backward needs it

  return logits
}

// Helper: tile a 1D bias [D] to [N, D] for broadcast add
async function tileBias(bias: Tensor, N: number): Promise<Tensor> {
  const D = bias.size
  const biasArr = await bias.toArray()
  const tiled = new Float32Array(N * D)
  for (let i = 0; i < N; i++) {
    tiled.set(biasArr, i * D)
  }
  const result = await Tensor.create(tiled, [N, D])
  // Propagate grad to bias via backward
  if (bias.requiresGrad) {
    // We need to record this for autograd
    const { recordOp, isGradEnabled } = await import('../gpu/autograd')
    if (isGradEnabled()) {
      recordOp(result, [bias], async (grad) => {
        // Sum grad along first axis: [N, D] -> [D]
        const gradArr = await grad.toArray()
        const summed = new Float32Array(D)
        for (let i = 0; i < N; i++) {
          for (let d = 0; d < D; d++) {
            summed[d] += gradArr[i * D + d]
          }
        }
        const gradBias = await Tensor.create(summed, [D])
        const { add: addOp } = await import('../gpu/ops')
        if (bias.grad) {
          const newGrad = await addOp(bias.grad, gradBias)
          bias.grad.dispose()
          gradBias.dispose()
          bias.grad = newGrad
        } else {
          bias.grad = gradBias
        }
      })
    }
  }
  return result
}

export function getAllParams(params: TransformerParams): Tensor[] {
  const all: Tensor[] = [
    params.tokenEmbed,
    params.posEmbed,
    params.lnFinalGamma,
    params.lnFinalBeta,
    params.lmHead,
  ]
  for (const layer of params.layers) {
    all.push(
      layer.lnAttnGamma, layer.lnAttnBeta,
      layer.Wq, layer.Wk, layer.Wv, layer.Wo,
      layer.lnFFGamma, layer.lnFFBeta,
      layer.W1, layer.b1, layer.W2, layer.b2,
    )
  }
  return all
}
