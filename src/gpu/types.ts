export type DType = 'f32'

export type Shape = number[]

export interface TensorSpec {
  shape: Shape
  dtype: DType
  size: number
}

export interface DeviceInfo {
  adapterName: string
  maxBufferSize: number
  maxStorageBufferBindingSize: number
  maxComputeWorkgroupSizeX: number
  hasF16: boolean
}
