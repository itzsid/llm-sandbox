// Sum reduction of array to scalar
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
  size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> wg_buf: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let tid = lid.x;

  // Each thread sums a strided chunk
  var sum = 0.0;
  for (var i = tid; i < params.size; i += 256u) {
    sum += input[i];
  }
  wg_buf[tid] = sum;
  workgroupBarrier();

  // Parallel reduction
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      wg_buf[tid] += wg_buf[tid + stride];
    }
    workgroupBarrier();
  }

  if (tid == 0u) {
    result[0] = wg_buf[0];
  }
}
