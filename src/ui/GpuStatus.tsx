import { useEffect, useState } from 'react'
import { getGPUDevice, getDeviceInfo } from '../gpu/device'
import type { DeviceInfo } from '../gpu/types'

export function GpuStatus() {
  const [info, setInfo] = useState<DeviceInfo | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getGPUDevice()
      .then(() => setInfo(getDeviceInfo()))
      .catch((e: Error) => setError(e.message))
  }, [])

  if (error) {
    return (
      <div className="gpu-status gpu-error">
        <span className="status-dot red" />
        <span>GPU: {error}</span>
        <a
          href="https://caniuse.com/webgpu"
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: 'var(--blue)', fontSize: '0.75rem', marginLeft: '0.5rem' }}
        >
          Check browser support
        </a>
      </div>
    )
  }

  if (!info) {
    return (
      <div className="gpu-status">
        <span className="status-dot yellow" />
        GPU: Initializing...
      </div>
    )
  }

  return (
    <div className="gpu-status">
      <span className="status-dot green" />
      GPU: {info.adapterName} | Buffer: {(info.maxBufferSize / 1024 / 1024).toFixed(0)}MB
      {info.hasF16 && ' | f16'}
    </div>
  )
}
