// Softmax along last dimension: one workgroup per row
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
  rows: u32,
  cols: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

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

  // Pass 1: find max (parallel reduction)
  var localMax = -3.402823e+38; // -FLT_MAX
  for (var i = tid; i < params.cols; i += 256u) {
    localMax = max(localMax, input[offset + i]);
  }
  wg_buf[tid] = localMax;
  workgroupBarrier();

  // Reduce to find row max
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      wg_buf[tid] = max(wg_buf[tid], wg_buf[tid + stride]);
    }
    workgroupBarrier();
  }
  let rowMax = wg_buf[0];
  workgroupBarrier();

  // Pass 2: compute exp(x - max) and sum
  var localSum = 0.0;
  for (var i = tid; i < params.cols; i += 256u) {
    let e = exp(input[offset + i] - rowMax);
    result[offset + i] = e;
    localSum += e;
  }
  wg_buf[tid] = localSum;
  workgroupBarrier();

  // Reduce to find sum
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      wg_buf[tid] += wg_buf[tid + stride];
    }
    workgroupBarrier();
  }
  let rowSum = wg_buf[0];
  workgroupBarrier();

  // Pass 3: normalize
  for (var i = tid; i < params.cols; i += 256u) {
    result[offset + i] /= rowSum;
  }
}
