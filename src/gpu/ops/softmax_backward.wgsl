// Softmax backward: dInput[i] = softmax[i] * (grad[i] - dot(grad, softmax))
@group(0) @binding(0) var<storage, read> softmaxOut: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params {
  rows: u32,
  cols: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> wg_buf: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let row = wid.x;
  if (row >= params.rows) { return; }
  let tid = lid.x;
  let offset = row * params.cols;

  // Compute dot(grad, softmax)
  var localDot = 0.0;
  for (var i = tid; i < params.cols; i += 256u) {
    localDot += grad[offset + i] * softmaxOut[offset + i];
  }
  wg_buf[tid] = localDot;
  workgroupBarrier();

  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { wg_buf[tid] += wg_buf[tid + stride]; }
    workgroupBarrier();
  }
  let dotProduct = wg_buf[0];
  workgroupBarrier();

  // Compute result
  for (var i = tid; i < params.cols; i += 256u) {
    result[offset + i] = softmaxOut[offset + i] * (grad[offset + i] - dotProduct);
  }
}
