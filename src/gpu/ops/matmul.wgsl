// Tiled matrix multiplication: C[M,N] = A[M,K] * B[K,N]
struct Params {
  M: u32,
  N: u32,
  K: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE_SIZE = 16u;

var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let row = wid.y * TILE_SIZE + lid.y;
  let col = wid.x * TILE_SIZE + lid.x;

  var sum = 0.0;
  let numTiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;

  for (var t = 0u; t < numTiles; t++) {
    let aCol = t * TILE_SIZE + lid.x;
    let bRow = t * TILE_SIZE + lid.y;

    if (row < params.M && aCol < params.K) {
      tileA[lid.y][lid.x] = a[row * params.K + aCol];
    } else {
      tileA[lid.y][lid.x] = 0.0;
    }

    if (bRow < params.K && col < params.N) {
      tileB[lid.y][lid.x] = b[bRow * params.N + col];
    } else {
      tileB[lid.y][lid.x] = 0.0;
    }

    workgroupBarrier();

    for (var k = 0u; k < TILE_SIZE; k++) {
      sum += tileA[lid.y][k] * tileB[k][lid.x];
    }

    workgroupBarrier();
  }

  if (row < params.M && col < params.N) {
    c[row * params.N + col] = sum;
  }
}
