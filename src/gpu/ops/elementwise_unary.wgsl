@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct Params {
  size: u32,
  op: u32, // 0=relu, 1=gelu, 2=tanh, 3=exp, 4=neg, 5=scalar_mul
  scalar: f32,
}
@group(0) @binding(2) var<uniform> params: Params;

fn gelu(x: f32) -> f32 {
  // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  let c = 0.7978845608; // sqrt(2/pi)
  let inner = c * (x + 0.044715 * x * x * x);
  return 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.size) { return; }
  let v = input[idx];
  switch params.op {
    case 0u: { result[idx] = max(v, 0.0); }      // relu
    case 1u: { result[idx] = gelu(v); }            // gelu
    case 2u: { result[idx] = tanh(v); }            // tanh
    case 3u: { result[idx] = exp(v); }              // exp
    case 4u: { result[idx] = -v; }                  // neg
    case 5u: { result[idx] = v * params.scalar; }   // scalar_mul
    default: { result[idx] = v; }
  }
}
