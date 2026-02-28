// Causal mask backward: zero gradient where mask was applied (col > row), pass through elsewhere
// Works for both 2D [T, T] and 3D [B, T, T] (flattened, rows/cols define the T*T slice)
@group(0) @binding(0) var<storage, read> grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
  totalSlices: u32,
  T: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let totalElements = params.totalSlices * params.T * params.T;
  let idx = gid.x;
  if (idx >= totalElements) { return; }

  let posInSlice = idx % (params.T * params.T);
  let row = posInSlice / params.T;
  let col = posInSlice % params.T;

  if (col > row) {
    result[idx] = 0.0;
  } else {
    result[idx] = grad[idx];
  }
}
