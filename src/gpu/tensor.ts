import type { DType, Shape } from './types'
import { getGPUDevice } from './device'

export type GradFn = (grad: Tensor) => Promise<void>

export class Tensor {
  buffer: GPUBuffer
  shape: Shape
  strides: number[]
  dtype: DType
  size: number

  requiresGrad: boolean
  grad: Tensor | null = null
  _gradFn: GradFn | null = null
  _parents: Tensor[] = []
  _disposed = false

  private constructor(
    buffer: GPUBuffer,
    shape: Shape,
    requiresGrad = false,
  ) {
    this.buffer = buffer
    this.shape = shape
    this.dtype = 'f32'
    this.size = shape.reduce((a, b) => a * b, 1)
    this.strides = Tensor.computeStrides(shape)
    this.requiresGrad = requiresGrad
  }

  static computeStrides(shape: Shape): number[] {
    const strides = new Array(shape.length)
    let stride = 1
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = stride
      stride *= shape[i]
    }
    return strides
  }

  static async create(
    data: Float32Array | number[],
    shape: Shape,
    opts?: { requiresGrad?: boolean },
  ): Promise<Tensor> {
    const device = await getGPUDevice()
    const arr = data instanceof Float32Array ? data : new Float32Array(data)
    const size = shape.reduce((a, b) => a * b, 1)
    if (arr.length !== size) {
      throw new Error(`Data length ${arr.length} doesn't match shape ${shape} (size ${size})`)
    }
    const buffer = device.createBuffer({
      size: arr.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    })
    new Float32Array(buffer.getMappedRange()).set(arr)
    buffer.unmap()
    return new Tensor(buffer, shape, opts?.requiresGrad ?? false)
  }

  static async empty(shape: Shape, opts?: { requiresGrad?: boolean }): Promise<Tensor> {
    const device = await getGPUDevice()
    const size = shape.reduce((a, b) => a * b, 1)
    const buffer = device.createBuffer({
      size: size * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    })
    return new Tensor(buffer, shape, opts?.requiresGrad ?? false)
  }

  static async zeros(shape: Shape, opts?: { requiresGrad?: boolean }): Promise<Tensor> {
    const size = shape.reduce((a, b) => a * b, 1)
    return Tensor.create(new Float32Array(size), shape, opts)
  }

  static async ones(shape: Shape, opts?: { requiresGrad?: boolean }): Promise<Tensor> {
    const size = shape.reduce((a, b) => a * b, 1)
    const data = new Float32Array(size).fill(1)
    return Tensor.create(data, shape, opts)
  }

  static async fromU32(data: Uint32Array, shape: Shape): Promise<Tensor> {
    const device = await getGPUDevice()
    const buffer = device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    })
    new Uint32Array(buffer.getMappedRange()).set(data)
    buffer.unmap()
    const t = new Tensor(buffer, shape, false)
    t.dtype = 'f32' // storage is u32, but we keep the type system simple
    return t
  }

  async toArray(): Promise<Float32Array> {
    const device = await getGPUDevice()
    const staging = device.createBuffer({
      size: this.size * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    })
    const encoder = device.createCommandEncoder()
    encoder.copyBufferToBuffer(this.buffer, 0, staging, 0, this.size * 4)
    device.queue.submit([encoder.finish()])
    await staging.mapAsync(GPUMapMode.READ)
    const data = new Float32Array(staging.getMappedRange()).slice()
    staging.unmap()
    staging.destroy()
    return data
  }

  dispose(): void {
    if (!this._disposed) {
      this.buffer.destroy()
      this._disposed = true
    }
    if (this.grad) {
      this.grad.dispose()
      this.grad = null
    }
  }
}
