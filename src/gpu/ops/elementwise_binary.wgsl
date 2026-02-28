@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params {
  size: u32,
  op: u32, // 0=add, 1=sub, 2=mul, 3=div
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.size) { return; }
  let va = a[idx];
  let vb = b[idx];
  switch params.op {
    case 0u: { result[idx] = va + vb; }
    case 1u: { result[idx] = va - vb; }
    case 2u: { result[idx] = va * vb; }
    case 3u: { result[idx] = va / vb; }
    default: { result[idx] = 0.0; }
  }
}
