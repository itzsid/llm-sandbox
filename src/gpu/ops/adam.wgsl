// Adam optimizer: update params in-place
@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read> grads: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;  // first moment
@group(0) @binding(3) var<storage, read_write> v: array<f32>;  // second moment

struct AdamParams {
  size: u32,
  lr: f32,
  beta1: f32,
  beta2: f32,
  eps: f32,
  beta1_corr: f32, // 1 - beta1^t
  beta2_corr: f32, // 1 - beta2^t
  weight_decay: f32,
}
@group(0) @binding(4) var<uniform> adam: AdamParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= adam.size) { return; }

  let g = grads[idx];

  // Update biased first moment (unmodified gradient)
  m[idx] = adam.beta1 * m[idx] + (1.0 - adam.beta1) * g;
  // Update biased second moment (unmodified gradient)
  v[idx] = adam.beta2 * v[idx] + (1.0 - adam.beta2) * g * g;

  // Bias-corrected estimates
  let m_hat = m[idx] / adam.beta1_corr;
  let v_hat = v[idx] / adam.beta2_corr;

  // Adam update + decoupled weight decay (true AdamW)
  params[idx] -= adam.lr * (m_hat / (sqrt(v_hat) + adam.eps) + adam.weight_decay * params[idx]);
}
