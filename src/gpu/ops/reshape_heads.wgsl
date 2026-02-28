// Reshape heads: bidirectional split/merge
// direction=0 (split): [B*T, D] -> [B*nHeads, T, headDim]
//   out[((b*nH+h)*T+t)*headDim + d] = in[(b*T+t)*D + h*headDim + d]
// direction=1 (merge): [B*nHeads, T, headDim] -> [B*T, D]
//   out[(b*T+t)*D + h*headDim + d] = in[((b*nH+h)*T+t)*headDim + d]
struct Params {
  B: u32,
  T: u32,
  nHeads: u32,
  headDim: u32,
  direction: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let totalElements = params.B * params.nHeads * params.T * params.headDim;
  let idx = gid.x;
  if (idx >= totalElements) { return; }

  let D = params.nHeads * params.headDim;

  // Decompose idx into (b, h, t, d) in the [B*nH, T, headDim] layout
  let d = idx % params.headDim;
  let rem = idx / params.headDim;
  let t = rem % params.T;
  let rem2 = rem / params.T;
  let h = rem2 % params.nHeads;
  let b = rem2 / params.nHeads;

  let headIdx = ((b * params.nHeads + h) * params.T + t) * params.headDim + d;
  let flatIdx = (b * params.T + t) * D + h * params.headDim + d;

  if (params.direction == 0u) {
    // split: flat -> head
    result[headIdx] = input[flatIdx];
  } else {
    // merge: head -> flat
    result[flatIdx] = input[headIdx];
  }
}
