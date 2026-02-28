// LayerNorm backward
// Given: input, gamma, grad_output, mean, rstd
// Compute: grad_input, grad_gamma, grad_beta
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> grad_output: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_gamma: array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_beta: array<f32>;

struct Params {
  rows: u32,
  cols: u32,
  eps: f32,
}
@group(0) @binding(6) var<uniform> params: Params;

var<workgroup> wg_buf: array<f32, 256>;
var<workgroup> wg_buf2: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let row = wid.x;
  if (row >= params.rows) { return; }
  let tid = lid.x;
  let offset = row * params.cols;

  // Recompute mean
  var localSum = 0.0;
  for (var i = tid; i < params.cols; i += 256u) {
    localSum += input[offset + i];
  }
  wg_buf[tid] = localSum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { wg_buf[tid] += wg_buf[tid + stride]; }
    workgroupBarrier();
  }
  let mean = wg_buf[0] / f32(params.cols);
  workgroupBarrier();

  // Recompute variance
  var localVar = 0.0;
  for (var i = tid; i < params.cols; i += 256u) {
    let diff = input[offset + i] - mean;
    localVar += diff * diff;
  }
  wg_buf[tid] = localVar;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { wg_buf[tid] += wg_buf[tid + stride]; }
    workgroupBarrier();
  }
  let variance = wg_buf[0] / f32(params.cols);
  let rstd = 1.0 / sqrt(variance + params.eps);
  workgroupBarrier();

  // grad_beta += grad_output (accumulated atomically per col - CPU does final accumulation)
  // grad_gamma += grad_output * normalized (same)
  // For now: this kernel computes grad_input per row,
  // and contributes to grad_gamma/grad_beta atomically would be ideal
  // but WGSL lacks f32 atomics. We'll accumulate on CPU side across rows.

  // Compute ds = sum(grad_output * gamma * (input - mean)) and db = sum(grad_output * gamma)
  var localDs = 0.0;
  var localDb = 0.0;
  for (var i = tid; i < params.cols; i += 256u) {
    let g = grad_output[offset + i] * gamma[i];
    localDs += g * (input[offset + i] - mean);
    localDb += g;
  }
  wg_buf[tid] = localDs;
  wg_buf2[tid] = localDb;
  workgroupBarrier();

  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      wg_buf[tid] += wg_buf[tid + stride];
      wg_buf2[tid] += wg_buf2[tid + stride];
    }
    workgroupBarrier();
  }
  let ds = wg_buf[0];
  let db = wg_buf2[0];
  workgroupBarrier();

  let colsF = f32(params.cols);
  for (var i = tid; i < params.cols; i += 256u) {
    let g = grad_output[offset + i] * gamma[i];
    let xhat = (input[offset + i] - mean) * rstd;
    grad_input[offset + i] = rstd * (g - (db + xhat * ds * rstd) / colsF);
  }
}
