// 2D transpose: result[j,i] = input[i,j]
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
  rows: u32,
  cols: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.rows * params.cols) { return; }
  let row = idx / params.cols;
  let col = idx % params.cols;
  result[col * params.rows + row] = input[idx];
}
