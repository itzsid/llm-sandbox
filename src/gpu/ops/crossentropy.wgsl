// Cross-entropy loss: log-softmax + NLL
// Input: logits [N, C], tgt_idxs [N] (u32 indices)
// Output: loss [1] (scalar mean)
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> tgt_idxs: array<u32>;
@group(0) @binding(2) var<storage, read_write> losses: array<f32>; // per-sample losses [N]

struct Params {
  N: u32,
  C: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.N) { return; }
  let offset = i * params.C;
  let tgt_idx = tgt_idxs[i];

  // Find max for numerical stability
  var maxVal = logits[offset];
  for (var c = 1u; c < params.C; c++) {
    maxVal = max(maxVal, logits[offset + c]);
  }

  // Compute log-sum-exp
  var sumExp = 0.0;
  for (var c = 0u; c < params.C; c++) {
    sumExp += exp(logits[offset + c] - maxVal);
  }
  let logSumExp = log(sumExp) + maxVal;

  // NLL = -(logits[tgt_idx] - logSumExp)
  losses[i] = -(logits[offset + tgt_idx] - logSumExp);
}
