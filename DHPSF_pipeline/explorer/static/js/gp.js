// Gaussian-process (kriging) interpolation of scattered 3D displacements.
//
// Each displacement component is modelled as a smooth random field with a squared-exponential
// covariance s² exp(-|p - q|² / 2ℓ²) and an unknown constant mean (ordinary kriging); every bead
// has its own measurement variance (its localization errors at both frames), so noisy beads -- and
// the noisy z axis -- count less and are smoothed rather than followed. The length scale ℓ and the
// amplitudes s² are chosen by maximizing the marginal likelihood over a grid. Pure functions, no DOM.

// in-place lower Cholesky of a dense symmetric n×n matrix (row-major Float64Array); false if not PD
export function cholesky(A, n) {
  for (let j = 0; j < n; j++) {
    let d = A[j * n + j];
    for (let k = 0; k < j; k++) d -= A[j * n + k] * A[j * n + k];
    if (!(d > 0)) return false;
    d = Math.sqrt(d);
    A[j * n + j] = d;
    for (let i = j + 1; i < n; i++) {
      let s = A[i * n + j];
      for (let k = 0; k < j; k++) s -= A[i * n + k] * A[j * n + k];
      A[i * n + j] = s / d;
    }
  }
  return true;
}
// solve L Lᵀ x = b
export function cholSolve(L, n, b) {
  const x = Float64Array.from(b);
  for (let i = 0; i < n; i++) { let s = x[i]; for (let k = 0; k < i; k++) s -= L[i * n + k] * x[k]; x[i] = s / L[i * n + i]; }
  for (let i = n - 1; i >= 0; i--) { let s = x[i]; for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k]; x[i] = s / L[i * n + i]; }
  return x;
}

function sqDists(P) {
  const n = P.length / 3, D = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) {
    const dx = P[3 * i] - P[3 * j], dy = P[3 * i + 1] - P[3 * j + 1], dz = P[3 * i + 2] - P[3 * j + 2];
    D[i * n + j] = D[j * n + i] = dx * dx + dy * dy + dz * dz;
  }
  return D;
}

// one component: factorize, ordinary-kriging mean, weights and log marginal likelihood
function krige(D, n, y, noise, ell, s2) {
  const A = new Float64Array(n * n), g = -0.5 / (ell * ell);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < i; j++) A[i * n + j] = A[j * n + i] = s2 * Math.exp(g * D[i * n + j]);
    A[i * n + i] = s2 + noise[i];
  }
  if (!cholesky(A, n)) return null;
  const ones = new Float64Array(n).fill(1);
  const a1 = cholSolve(A, n, ones), ay = cholSolve(A, n, y);
  let s1 = 0, sy = 0;
  for (let i = 0; i < n; i++) { s1 += a1[i]; sy += ay[i]; }
  const mu = sy / s1;
  const r = y.map((v) => v - mu);
  const alpha = cholSolve(A, n, r);
  let quad = 0, logdet = 0;
  for (let i = 0; i < n; i++) { quad += r[i] * alpha[i]; logdet += Math.log(A[i * n + i]); }
  const lml = -0.5 * quad - logdet - 0.5 * n * Math.log(2 * Math.PI);
  return { mu, alpha, lml };
}

function median(a) { const s = Float64Array.from(a).sort(); return s.length ? s[s.length >> 1] : NaN; }
function variance(a) { let m = 0; for (const v of a) m += v; m /= a.length; let s = 0; for (const v of a) s += (v - m) ** 2; return s / Math.max(1, a.length - 1); }

export function nearestSpacing(P) {
  const n = P.length / 3, D = sqDists(P), nn = [];
  for (let i = 0; i < n; i++) { let m = Infinity; for (let j = 0; j < n; j++) if (j !== i && D[i * n + j] < m) m = D[i * n + j]; nn.push(Math.sqrt(m)); }
  return median(nn);
}

// Choose ℓ (shared by the three components) and s² (per component) by marginal likelihood.
// P: Float64Array 3n positions (µm); U: [ux, uy, uz] Float64Arrays; V: matching noise variances.
export function fitHyper(P, U, V, fixedEll = null) {
  const it = fitHyperSteps(P, U, V, fixedEll);
  let r;
  while (!(r = it.next()).done);
  return r.value;
}

// the same, as a generator yielding the fraction done after each length scale (for a progress bar)
export function* fitHyperSteps(P, U, V, fixedEll = null) {
  const n = P.length / 3, D = sqDists(P);
  const nn = nearestSpacing(P);
  const ells = fixedEll ? [fixedEll] : [0.5, 0.7, 1, 1.4, 2, 2.8, 4, 5.6, 8].map((f) => f * nn);
  let best = null;
  for (const [i, ell] of ells.entries()) {
    if (i) yield i / ells.length;
    let total = 0; const s2s = [];
    for (let c = 0; c < 3; c++) {
      const v0 = Math.max(variance(U[c]) - median(V[c]), 1e-6 * (1 + median(V[c])));
      let bc = null;
      for (const f of [0.1, 0.3, 1, 3, 10]) {
        const k = krige(D, n, U[c], V[c], ell, f * v0);
        if (k && (!bc || k.lml > bc.lml)) bc = { lml: k.lml, s2: f * v0 };
      }
      if (!bc) { total = -Infinity; break; }
      total += bc.lml; s2s.push(bc.s2);
    }
    if (!best || total > best.lml) best = { ell, s2: s2s, lml: total };
  }
  return { ...best, nn };
}

// Condition on the data with given hyperparameters; returns a predictor over many points.
export function condition(P, U, V, ell, s2) {
  const n = P.length / 3, D = sqDists(P);
  const comps = [0, 1, 2].map((c) => krige(D, n, U[c], V[c], ell, s2[c]));
  if (comps.some((c) => !c)) return null;
  const g = -0.5 / (ell * ell);
  // Q: Float64Array 3m query points -> { u: [3 Float64Array m], density: Float64Array m, grad? }
  // density = Σ exp(-d²/2ℓ²): about the number of beads within ~ℓ (to hide areas without data).
  // withGrad: also the exact derivatives of the interpolated field, grad[3c + j] = ∂u_c/∂x_j
  // (the kernel's derivative: ∂/∂x_j exp(-d²/2ℓ²) = -(x_j - p_j)/ℓ² exp(-d²/2ℓ²)).
  const al = comps.map((c, k) => c.alpha.map((v) => v * s2[k]));
  const il2 = 1 / (ell * ell);
  return (Q, withGrad = false) => {
    const m = Q.length / 3;
    const out = [new Float64Array(m), new Float64Array(m), new Float64Array(m)], dens = new Float64Array(m);
    const grad = withGrad ? Array.from({ length: 9 }, () => new Float64Array(m)) : null;
    const gq = new Float64Array(9);
    for (let q = 0; q < m; q++) {
      const x = Q[3 * q], y = Q[3 * q + 1], z = Q[3 * q + 2];
      let a0 = 0, a1 = 0, a2 = 0, w = 0;
      if (withGrad) gq.fill(0);
      for (let i = 0; i < n; i++) {
        const dx = x - P[3 * i], dy = y - P[3 * i + 1], dz = z - P[3 * i + 2];
        const e = Math.exp(g * (dx * dx + dy * dy + dz * dz));
        const e0 = e * al[0][i], e1 = e * al[1][i], e2 = e * al[2][i];
        a0 += e0; a1 += e1; a2 += e2; w += e;
        if (withGrad) {
          gq[0] -= e0 * dx; gq[1] -= e0 * dy; gq[2] -= e0 * dz;
          gq[3] -= e1 * dx; gq[4] -= e1 * dy; gq[5] -= e1 * dz;
          gq[6] -= e2 * dx; gq[7] -= e2 * dy; gq[8] -= e2 * dz;
        }
      }
      out[0][q] = comps[0].mu + a0; out[1][q] = comps[1].mu + a1; out[2][q] = comps[2].mu + a2;
      dens[q] = w;
      if (withGrad) for (let k = 0; k < 9; k++) grad[k][q] = gq[k] * il2;
    }
    return { u: out, density: dens, grad, mean: comps.map((c) => c.mu) };
  };
}
