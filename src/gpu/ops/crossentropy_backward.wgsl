// Cross-entropy backward: dLogits = softmax(logits) - one_hot(tgt_idx)
// Then scale by upstream grad / N
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> tgt_idxs: array<u32>;
@group(0) @binding(2) var<storage, read> grad: array<f32>; // scalar [1]
@group(0) @binding(3) var<storage, read_write> grad_logits: array<f32>;

struct Params {
  N: u32,
  C: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.N) { return; }
  let offset = i * params.C;
  let tgt_idx = tgt_idxs[i];
  let scale = grad[0] / f32(params.N);

  // Compute softmax for this sample
  var maxVal = logits[offset];
  for (var c = 1u; c < params.C; c++) {
    maxVal = max(maxVal, logits[offset + c]);
  }

  var sumExp = 0.0;
  for (var c = 0u; c < params.C; c++) {
    sumExp += exp(logits[offset + c] - maxVal);
  }

  for (var c = 0u; c < params.C; c++) {
    var sm = exp(logits[offset + c] - maxVal) / sumExp;
    if (c == tgt_idx) {
      sm -= 1.0;
    }
    grad_logits[offset + c] = sm * scale;
  }
}
