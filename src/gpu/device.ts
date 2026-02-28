import type { DeviceInfo } from './types'

let device: GPUDevice | null = null
let deviceInfo: DeviceInfo | null = null

export async function getGPUDevice(): Promise<GPUDevice> {
  if (device) return device

  if (!navigator.gpu) {
    throw new Error('WebGPU not supported in this browser')
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  })
  if (!adapter) {
    throw new Error('No WebGPU adapter found')
  }

  const hasF16 = adapter.features.has('shader-f16')
  const requiredFeatures: GPUFeatureName[] = []
  if (hasF16) requiredFeatures.push('shader-f16')

  const adapterLimits = adapter.limits
  device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      maxBufferSize: Math.min(256 * 1024 * 1024, adapterLimits.maxBufferSize),
      maxStorageBufferBindingSize: Math.min(
        256 * 1024 * 1024,
        adapterLimits.maxStorageBufferBindingSize,
      ),
    },
  })

  device.lost.then((info) => {
    console.error('WebGPU device lost:', info.message)
    device = null
    deviceInfo = null
  })

  deviceInfo = {
    adapterName: adapter.info?.device || 'Unknown GPU',
    maxBufferSize: device.limits.maxBufferSize,
    maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
    maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
    hasF16,
  }

  return device
}

export function getDeviceInfo(): DeviceInfo | null {
  return deviceInfo
}

export function destroyDevice(): void {
  if (device) {
    device.destroy()
    device = null
    deviceInfo = null
  }
}
