// Layer normalization: one workgroup per row
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;

struct Params {
  rows: u32,
  cols: u32,
  eps: f32,
}
@group(0) @binding(4) var<uniform> params: Params;

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

  // Compute mean
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

  // Compute variance
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

  // Normalize + scale + shift
  for (var i = tid; i < params.cols; i += 256u) {
    let normalized = (input[offset + i] - mean) * rstd;
    result[offset + i] = normalized * gamma[i] + beta[i];
  }
}
