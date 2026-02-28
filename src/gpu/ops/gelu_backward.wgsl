// GELU backward: dx = grad * gelu'(x)
// gelu'(x) = 0.5*(1+tanh(c*(x+0.044715*x^3))) + 0.5*x*sech²(c*(x+0.044715*x^3))*c*(1+3*0.044715*x²)
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params {
  size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.size) { return; }
  let v = x[idx];
  let c = 0.7978845608; // sqrt(2/pi)
  let inner = c * (v + 0.044715 * v * v * v);
  let t = tanh(inner);
  let sech2 = 1.0 - t * t;
  let dInner = c * (1.0 + 3.0 * 0.044715 * v * v);
  let geluGrad = 0.5 * (1.0 + t) + 0.5 * v * sech2 * dInner;
  result[idx] = grad[idx] * geluGrad;
}
