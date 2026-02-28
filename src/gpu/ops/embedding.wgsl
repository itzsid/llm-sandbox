// Embedding lookup: gather from [V, D] table using u32 indices -> [N, D] output
@group(0) @binding(0) var<storage, read> table: array<f32>;   // [V * D]
@group(0) @binding(1) var<storage, read> indices: array<u32>;  // [N]
@group(0) @binding(2) var<storage, read_write> result: array<f32>; // [N * D]

struct Params {
  N: u32,     // number of tokens
  D: u32,     // embedding dimension
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.N * params.D) { return; }
  let tokenIdx = idx / params.D;
  let dimIdx = idx % params.D;
  let vocabIdx = indices[tokenIdx];
  result[idx] = table[vocabIdx * params.D + dimIdx];
}
