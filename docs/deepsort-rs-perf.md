# deepsort-rs — rendre l'association rapide quand il y a du monde

Changements à faire dans le dépôt `deepsort-rs` (révision actuelle
`6f12a5d`), puis à rebrancher dans VisionCam. Mesures de départ dans
`docs/performance.md`, section « Tracker Rust à 4 caméras ».

## Le problème

Association seule, personnes synthétiques, 300 images, `nn_budget=100` :

| Personnes par image | deep_sort_realtime | deepsort-rs |
|---:|---:|---:|
| 3 | 0,69 ms | 0,26 ms |
| 10 | 2,14 ms | 2,65 ms |
| 40 | 9,74 ms | 43,12 ms |

Sur MOT17-04 (détections publiques, `tools.eval_mot`), pistes identiques
(MOTA 73,6 %, IDF1 72,3 %, 102 IDs) mais 138–145 ms par image contre
40–42 ms.

Le coût vient des distances d'apparence de la cascade : pistes × détections
× échantillons (≤ 100) × 512 multiplications-additions, soit ~82 millions par
image à 40 personnes.

- **Référence Python** (`nn_matching.NearestNeighborDistanceMetric`) : pour
  chaque piste, un produit matriciel `samples (n×512) · features (m×512)ᵀ`
  par numpy, donc BLAS (SIMD, FMA, plusieurs accumulateurs), puis le minimum
  par colonne.
- **deepsort-rs** (`src/metrics.rs`, `nn_cosine_distance` →
  `cosine_distance_normalized`) : un produit scalaire par paire
  (échantillon, détection), `a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>()`.
  1. La somme flottante est une chaîne de dépendances : sans réassociation
     permise, le compilateur ne la vectorise pas et chaque addition attend la
     précédente (~4 cycles de latence).
  2. Les échantillons sont des `Vec<Vec<f64>>` (`Track::features`,
     `Tracker::feature_bank`) : deux fois plus d'octets qu'en `f32`, une
     allocation par vecteur, pas de mémoire contiguë.
  3. Pas de blocage en tuiles : chaque échantillon est relu pour chaque
     détection.

S'y ajoute un point qui pèse en multi-caméras : `PyTracker::update`
(`src/lib.rs`) garde le GIL pendant tout le calcul (pas de `py.detach`). À
40 personnes, les autres threads Python attendent 43 ms.

## Changements

### 1. Échantillons d'apparence en `f32` contigus

Nouveau type, par exemple dans `src/metrics.rs` :

```rust
/// Échantillons d'une piste, ligne par ligne : `data[i*dim..(i+1)*dim]`.
pub struct Samples {
    pub dim: usize,
    pub data: Vec<f32>,
}

impl Samples {
    pub fn len(&self) -> usize { self.data.len() / self.dim }
    pub fn push(&mut self, v: &[f32]) { self.data.extend_from_slice(v); }
    /// Garde les `budget` derniers, comme `entry.drain(0..excess)` aujourd'hui.
    pub fn keep_last(&mut self, budget: usize) {
        let n = self.len();
        if n > budget {
            self.data.drain(0..(n - budget) * self.dim);
        }
    }
    pub fn last(&self) -> &[f32] { &self.data[self.data.len() - self.dim..] }
}
```

- `Tracker::feature_bank` : `HashMap<u64, Samples>` au lieu de
  `HashMap<u64, Vec<Vec<f64>>>`.
- `Track::features` : `Samples` aussi (ou `Vec<Vec<f32>>`, peu importe, il
  ne contient que quelques vecteurs entre deux `flush_features`).
- `Tracker::update` : les embeddings normalisés deviennent une matrice
  `m × dim` en `f32` contigu (`Vec<f32>`), au lieu de `Vec<Vec<f64>>`.
  `normalize` passe en `f32` (accumuler la norme en `f64` ne coûte rien et
  reste plus précis).
- `flush_features` : `append` → copie des lignes, `drain` → `keep_last`,
  `entry.last().clone()` → `last().to_vec()`. L'ordre d'éviction ne change
  pas (et le minimum des distances ne dépend pas de l'ordre des
  échantillons).
- `embeddings_from_numpy` (`src/lib.rs`) : lire directement le tableau
  `f32` contigu au lieu de le convertir en `Vec<Vec<f64>>`.

### 2. Distances par produit matriciel

Dans `matching_cascade` (`src/cascade.rs`), à chaque niveau :

1. Rassembler une fois les embeddings des `unmatched_detections` dans une
   matrice `Q` (`m × dim`, contiguë).
2. Pour chaque piste du niveau, avec ses échantillons `S` (`n × dim`) :
   `C = S · Qᵀ` (`n × m`) en un appel `sgemm`, puis
   `cost[j] = 1 - max_i C[i][j]` (le minimum de la distance est le maximum
   de la similarité).

Avec la crate `matrixmultiply` (déjà dans `Cargo.lock`, tirée par nalgebra ;
à déclarer en dépendance directe, `matrixmultiply = "0.3"`) :

```rust
/// cost[j] = min_i (1 - <S_i, Q_j>), S : n×dim, Q : m×dim, lignes normées.
pub fn nn_cosine_costs(s: &Samples, q: &[f32], m: usize, out: &mut Vec<f64>) {
    let (n, d) = (s.len(), s.dim);
    let mut c = vec![0f32; n * m];
    unsafe {
        // C (n×m) = S (n×d) · Qᵀ (d×m) ; Qᵀ s'obtient par les strides,
        // sans copie : ligne-stride 1, colonne-stride d.
        matrixmultiply::sgemm(
            n, d, m,
            1.0,
            s.data.as_ptr(), d as isize, 1,
            q.as_ptr(), 1, d as isize,
            0.0,
            c.as_mut_ptr(), m as isize, 1,
        );
    }
    out.clear();
    out.extend((0..m).map(|j| {
        let best = (0..n).map(|i| c[i * m + j]).fold(f32::NEG_INFINITY, f32::max);
        1.0 - best as f64
    }));
}
```

La matrice de coût reste en `f64` en aval (gating de Kalman, hongrois) : rien
d'autre ne change dans `gate_cost_matrix` ni `min_cost_matching`.

Si la parité casse à cause du `f32` (cf. Validation), repli : garder les
données en `f64` contigu et appeler `matrixmultiply::dgemm`. Moitié moins de
débit qu'en `f32`, mais toujours vectorisé : le gros du gain vient de la
contiguïté et du produit matriciel, pas de la précision.

### 3. Relâcher le GIL pendant `update`

Dans `PyTracker::update` (`src/lib.rs`), une fois les tableaux numpy copiés
en mémoire Rust (ce que font déjà `boxes_from_numpy` et
`embeddings_from_numpy`) :

```rust
let inner = &mut self.inner;
let tracks = py.detach(|| inner.update(&boxes, &embeddings));
```

(`py.detach` est le nom de `allow_threads` depuis PyO3 0.26 ; le projet est en
0.29.) `RustTracker` ne contient que des `Vec`, `HashMap` et des matrices
nalgebra : il est `Send`, la closure compile telle quelle. La construction du
tableau de sortie reste sous le GIL.

### 4. Hors périmètre

- L'algorithme hongrois (`pathfinding::kuhn_munkres_min`), le Kalman et le
  passage IoU : négligeables à ces tailles, ne pas y toucher dans ce commit.
- Pas de `-ffast-math` ni de réduction manuelle à plusieurs accumulateurs :
  `sgemm` les fait déjà, proprement.

## Validation

Dans `deepsort-rs` :

1. `cargo test` et `cargo clippy --all-targets -- -D warnings`.
2. `maturin develop --release`, puis `pytest tests/` :
   `tests/test_parity.py` doit rester vert (mêmes IDs, boîtes à 1e-4 de
   `deep_sort_realtime`). C'est le test qui dira si le passage en `f32`
   est acceptable.
3. `scripts/bench.py` avant/après, en release.

Dans VisionCam, après avoir pointé sur la nouvelle révision :

1. `pyproject.toml` : nouveau `rev` dans `[tool.uv.sources]` (`deepsort-rs`),
   puis `uv lock --upgrade-package deepsort-rs && uv sync`.
2. Banc d'association (script ci-dessous) : à 40 personnes, viser sous les
   9,74 ms de deep_sort_realtime ; à 3, ne pas régresser (0,26 ms).
3. Qualité inchangée sur MOT17-04 :

   ```bash
   uv run -m tools.eval_mot --seq ~/datasets/MOT17/train/MOT17-04-SDP \
       --variant rust:100 --variant python:100
   ```

   Référence : MOTA 73,6 %, IDF1 72,3 %, 102 IDs, FP 3 204, FN 9 247.
   Le temps par image du Rust doit passer sous celui du Python (40–42 ms,
   embedder compris).
4. 4 caméras dans un processus, `TRACKER_BACKEND=rust` (même protocole que
   `docs/performance.md`) : référence 96,4 i/s au total avant changement.
5. Consigner les chiffres dans `docs/performance.md` et `CHANGELOG.md` de
   deepsort-rs. Commit `perf(association): ...` côté crate,
   `chore(deps): bump deepsort-rs to <rev>` côté VisionCam.

### Banc d'association (depuis la racine de VisionCam)

```python
"""Association-only timing, Python vs Rust, for k synthetic walkers."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, time
import numpy as np
sys.path.insert(0, os.getcwd())
import config
import deepsort_rs
from core.tracker_backends import new_deep_sort

FRAMES = 300
rng = np.random.default_rng(0)
for k in (3, 10, 40):
    pos = rng.uniform(0, 1500, (k, 2)); vel = rng.normal(0, 2, (k, 2))
    ident = rng.normal(size=(k, 512)); ident /= np.linalg.norm(ident, axis=1, keepdims=True)
    frames = []
    for _ in range(FRAMES):
        pos += vel
        e = ident + 0.05 * rng.normal(size=ident.shape); e /= np.linalg.norm(e, axis=1, keepdims=True)
        frames.append((pos.copy(), e.astype(np.float32)))
    py = new_deep_sort()
    rs = deepsort_rs.Tracker(max_age=config.DEEPSORT_MAX_AGE, n_init=config.DEEPSORT_N_INIT,
                             max_cosine_distance=config.DEEPSORT_MAX_COSINE_DISTANCE,
                             nn_budget=config.DEEPSORT_NN_BUDGET,
                             max_iou_distance=config.DEEPSORT_MAX_IOU_DISTANCE)
    t_py = t_rs = 0.0
    for p, e in frames:
        dets = [([x, y, 60.0, 160.0], 0.9, "person", None) for x, y in p]
        t = time.perf_counter(); py.update_tracks(dets, embeds=list(e.astype(np.float16))); t_py += time.perf_counter() - t
        boxes = np.array([[x, y, x + 60, y + 160] for x, y in p], dtype=np.float32)
        t = time.perf_counter(); rs.update(boxes, [0.9] * k, e); t_rs += time.perf_counter() - t
    print(f"k={k:2d}  python {1000*t_py/FRAMES:6.2f} ms/frame   rust {1000*t_rs/FRAMES:6.2f} ms/frame")
```

`uv run python bench_assoc.py` (le fichier où tu le colles).
