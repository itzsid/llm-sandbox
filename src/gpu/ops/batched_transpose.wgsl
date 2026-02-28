// Batched 2D transpose: result[b, j, i] = input[b, i, j]
// input: [batchSize, rows, cols] -> result: [batchSize, cols, rows]
struct Params {
  batchSize: u32,
  rows: u32,
  cols: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = gid.y;
  if (batch >= params.batchSize) { return; }

  let idx = gid.x;
  let sliceSize = params.rows * params.cols;
  if (idx >= sliceSize) { return; }

  let i = idx / params.cols;
  let j = idx % params.cols;

  let inOffset = batch * sliceSize + i * params.cols + j;
  let outOffset = batch * sliceSize + j * params.rows + i;
  result[outOffset] = input[inOffset];
}
