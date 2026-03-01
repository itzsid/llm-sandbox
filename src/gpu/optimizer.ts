import { getGPUDevice } from './device'
import { Tensor } from './tensor'
import { zeroGrad } from './autograd'

import adamShader from './ops/adam.wgsl'

interface AdamConfig {
  lr: number
  beta1: number
  beta2: number
  eps: number
  weightDecay: number
}

const DEFAULT_CONFIG: AdamConfig = {
  lr: 6e-4,
  beta1: 0.9,
  beta2: 0.95,
  eps: 1e-8,
  weightDecay: 0.1,
}

export interface ParamGroup {
  params: Tensor[]
  weightDecay: number
}

interface ParamState {
  m: GPUBuffer // first moment
  v: GPUBuffer // second moment
}

let pipeline: GPUComputePipeline | null = null

async function getPipeline(device: GPUDevice): Promise<GPUComputePipeline> {
  if (!pipeline) {
    const module = device.createShaderModule({ code: adamShader })
    pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    })
  }
  return pipeline
}

export class AdamOptimizer {
  private groups: ParamGroup[]
  private config: AdamConfig
  private states: Map<Tensor, ParamState> = new Map()
  private step_ = 0

  constructor(groups: ParamGroup[], config?: Partial<AdamConfig>) {
    this.groups = groups
    this.config = { ...DEFAULT_CONFIG, ...config }
  }

  get allParams(): Tensor[] {
    return this.groups.flatMap((g) => g.params)
  }

  setLR(lr: number): void {
    this.config.lr = lr
  }

  setStep(step: number): void {
    this.step_ = step
  }

  async step(): Promise<void> {
    this.step_++
    const device = await getGPUDevice()
    const pipe = await getPipeline(device)

    const beta1Corr = 1 - Math.pow(this.config.beta1, this.step_)
    const beta2Corr = 1 - Math.pow(this.config.beta2, this.step_)

    for (const group of this.groups) {
      for (const param of group.params) {
        if (!param.grad) continue

        let state = this.states.get(param)
        if (!state) {
          state = {
            m: device.createBuffer({
              size: param.size * 4,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
              mappedAtCreation: true,
            }),
            v: device.createBuffer({
              size: param.size * 4,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
              mappedAtCreation: true,
            }),
          }
          // Zero-init
          new Float32Array(state.m.getMappedRange()).fill(0)
          state.m.unmap()
          new Float32Array(state.v.getMappedRange()).fill(0)
          state.v.unmap()
          this.states.set(param, state)
        }

        // Pack uniform
        const uniformData = new ArrayBuffer(32)
        const u32 = new Uint32Array(uniformData)
        const f32 = new Float32Array(uniformData)
        u32[0] = param.size
        f32[1] = this.config.lr
        f32[2] = this.config.beta1
        f32[3] = this.config.beta2
        f32[4] = this.config.eps
        f32[5] = beta1Corr
        f32[6] = beta2Corr
        f32[7] = group.weightDecay // per-group weight decay

        const uniform = device.createBuffer({
          size: 32,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          mappedAtCreation: true,
        })
        new Uint8Array(uniform.getMappedRange()).set(new Uint8Array(uniformData))
        uniform.unmap()

        const entries: GPUBindGroupEntry[] = [
          { binding: 0, resource: { buffer: param.buffer } },
          { binding: 1, resource: { buffer: param.grad.buffer } },
          { binding: 2, resource: { buffer: state.m } },
          { binding: 3, resource: { buffer: state.v } },
          { binding: 4, resource: { buffer: uniform } },
        ]

        const bindGroup = device.createBindGroup({
          layout: pipe.getBindGroupLayout(0),
          entries,
        })

        const encoder = device.createCommandEncoder()
        const pass = encoder.beginComputePass()
        pass.setPipeline(pipe)
        pass.setBindGroup(0, bindGroup)
        pass.dispatchWorkgroups(Math.ceil(param.size / 256))
        pass.end()
        device.queue.submit([encoder.finish()])
        await device.queue.onSubmittedWorkDone()
        uniform.destroy()
      }
    }
  }

  zeroGrad(): void {
    zeroGrad(this.allParams)
  }

  dispose(): void {
    for (const state of this.states.values()) {
      state.m.destroy()
      state.v.destroy()
    }
    this.states.clear()
  }
}
