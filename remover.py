# ──────────────────────────────────────────────────────────────────────────────
# Universal Stripe Remover  (2-D, Multi-Direction)
#
# Variational model solved via PDHGMp
# (Primal-Dual Hybrid Gradient Method with dual extrapolation):
#
#   argmin  μ₁‖∇u‖_{2,1}
#         + Σᵢ ‖D_θᵢ sᵢ‖₁
#         + μ₂ Σᵢ ‖sᵢ‖₁
#   s.t.  u + Σᵢ sᵢ = F,   u ∈ [0, 1]
#
# Five stripe directions are optimised simultaneously:
#   0 : Vertical   (Δrow=1, Δcol= 0)
#   1 : 26.6° Left (Δrow=2, Δcol=+1)
#   2 : 45°  Left  (Δrow=1, Δcol=+1)
#   3 : 26.6° Right(Δrow=2, Δcol=−1)
#   4 : 45°  Right (Δrow=1, Δcol=−1)
# ──────────────────────────────────────────────────────────────────────────────

import math
import torch
import torch.nn.functional as TF

# number of stripe directions
_N = 5
# total primal variables: 1 (clean) + _N (stripes)
_N_VARS = 1 + _N


class UniversalStripeRemover:
    """Remove stripes from a 2-D image in five directions at once.

    Parameters
    ----------
    mu1 : float
        Smoothness weight (TV penalty on the clean image *u*).
    mu2 : float
        Sparsity weight on each stripe component *sᵢ*.
    device : torch.device | str | None
        Computation device.  ``None`` → auto-select CUDA / CPU.
    """

    def __init__(self, mu1: float = 0.33, mu2: float = 0.003, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mu1 = mu1
        self.mu2 = mu2
        # symmetric step sizes (τ = σ)
        self.tau = 0.35
        self.sigma = 0.35

    # ── public API ────────────────────────────────────────────────────────

    def process(self, image, iterations=500, tol=1e-5,
                proj=True, verbose=True):
        """De-stripe a single image **or** a pre-batched tensor.

        Parameters
        ----------
        image : array-like
            ``(H, W)`` or ``(B, H, W)``.
        iterations : int
            Maximum PDHGMp iterations.
        tol : float
            Relative-change threshold for early stopping (checked every 20 it).
        proj : bool
            Clamp the clean image to [0, 1].
        verbose : bool
            Print iteration progress.

        Returns
        -------
        torch.Tensor   on CPU, same shape as *image*.
        """
        x = self._to_tensor(image)
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)

        result = self._solve(x, iterations, tol, proj, verbose)

        return result.squeeze(0) if squeeze else result

    def process_tiled(self, image, n=1, iterations=500, tol=1e-5,
                      overlap=64, proj=True, verbose=True):
        """Tile → batch-process → reassemble with cosine blending.

        Parameters
        ----------
        image : array-like
            ``(H, W)`` single image.
        n : int
            Split into *n × n* tiles.  ``n = 1`` skips tiling.
        overlap : int
            Overlap margin (pixels) between adjacent tiles.
        *other* : same as :meth:`process`.

        Returns
        -------
        torch.Tensor   ``(H, W)`` on CPU.
        """
        data = self._to_tensor(image)
        if data.dim() == 3:
            data = data.squeeze(0)
        assert data.dim() == 2, 'Tiled processing requires a 2-D image.'

        if n <= 1:
            return self.process(data, iterations, tol, proj, verbose)

        H, W = data.shape

        # ── pad so dimensions are divisible by n ──
        data = self._reflect_pad(data,
                                 pad_bottom=(n - H % n) % n,
                                 pad_right=(n - W % n) % n)
        Hp, Wp = data.shape
        th_core, tw_core = Hp // n, Wp // n

        # ── choose safe overlap ──
        ov = min(overlap, th_core // 4, tw_core // 4)
        ov = max(ov, 0)

        # ── pad borders for edge tiles ──
        data = self._reflect_pad(data, ov, ov, ov, ov)

        th, tw = th_core + 2 * ov, tw_core + 2 * ov

        # ── extract & stack tiles ──
        tiles = [
            data[i * th_core: i * th_core + th,
                 j * tw_core: j * tw_core + tw]
            for i in range(n) for j in range(n)
        ]
        batch = torch.stack(tiles)
        if verbose:
            print('Tiling {}×{}: {} tiles of {}×{}, overlap={}'.format(
                n, n, n * n, th, tw, ov))

        # ── process batch ──
        results = self.process(batch, iterations, tol, proj, verbose)

        # ── reassemble with cosine blending ──
        weight = self._cosine_window(th, tw, ov)
        canvas = torch.zeros(Hp + 2 * ov, Wp + 2 * ov)
        wsum   = torch.zeros_like(canvas)

        for idx, (i, j) in enumerate((i, j) for i in range(n) for j in range(n)):
            y, x = i * th_core, j * tw_core
            canvas[y:y + th, x:x + tw] += results[idx] * weight
            wsum  [y:y + th, x:x + tw] += weight

        canvas /= wsum.clamp(min=1e-9)
        return canvas[ov:ov + Hp, ov:ov + Wp][:H, :W]

    # ── core solver ───────────────────────────────────────────────────────

    def _solve(self, data, iterations, tol, proj, verbose):
        """Run PDHGMp on *data* ``(B, H, W)``  →  clean image on CPU."""
        data = data.to(self.device, dtype=torch.float32)
        ts   = self.tau * self.sigma          # τσ
        lam  = self.mu1 / self.sigma          # TV projection radius
        qc   = 1.0 / self.sigma              # directional clip
        rc   = self.mu2 / self.sigma          # sparsity clip
        eps  = 1e-9

        # primal variables
        u = data.clone()
        s = [torch.zeros_like(data) for _ in range(_N)]

        # dual variables  (value, extrapolated)
        ph,  ph_ = self._zeros2(data)   # smoothness – horizontal
        pv,  pv_ = self._zeros2(data)   # smoothness – vertical
        q,   q_  = zip(*(self._zeros2(data) for _ in range(_N)))  # directional
        r,   r_  = zip(*(self._zeros2(data) for _ in range(_N)))  # sparsity
        q, q_ = list(q), list(q_)
        r, r_ = list(r), list(r_)

        prev_u = u.clone()
        buf = torch.empty_like(data)    # reusable scratch buffer

        with torch.no_grad():
            for k in range(iterations):
                if verbose:
                    print('\rIteration: {} / {}'.format(k + 1, iterations), end='')

                # ── 1. primal descent ─────────────────────────────────
                self._adj_grad(u, ph_, pv_, ts)
                for i in range(_N):
                    self._adj_dir(s[i], q_[i], i, ts)
                    s[i].sub_(r_[i], alpha=ts)

                # ── 2. constraint projection  u + Σsᵢ = F ────────────
                buf.copy_(data)
                for si in s:
                    buf.sub_(si)
                buf.sub_(u).div_(_N_VARS)
                u.add_(buf)
                for si in s:
                    si.add_(buf)

                # ── 3. box projection  u ∈ [0,1] ─────────────────────
                if proj:
                    # buf = under + over  (reuse buffer)
                    torch.clamp(u, max=0, out=buf)
                    buf.add_((u - 1).clamp_(min=0))
                    buf.div_(_N)
                    for si in s:
                        si.add_(buf)
                    u.clamp_(0, 1)

                # ── 4. dual update (Moreau / projection) ──────────────
                #  smoothness  –  isotropic TV (coupled projection)
                ph_.copy_(ph)
                pv_.copy_(pv)
                zh = ph + self._fwd(u, 1)
                zv = pv + self._fwd(u, 2)
                norm = torch.sqrt(zh * zh + zv * zv).clamp_(min=eps)
                scale = (lam / norm).clamp_(max=1.0)
                ph = zh.mul_(scale)
                pv = zv.mul_(scale)
                ph_.mul_(-1).add_(ph, alpha=2)
                pv_.mul_(-1).add_(pv, alpha=2)

                #  directional + sparsity
                for i in range(_N):
                    q_[i].copy_(q[i])
                    q[i] = (q[i] + self._dir_diff(s[i], i)).clamp_(-qc, qc)
                    q_[i].mul_(-1).add_(q[i], alpha=2)

                    r_[i].copy_(r[i])
                    r[i] = (r[i] + s[i]).clamp_(-rc, rc)
                    r_[i].mul_(-1).add_(r[i], alpha=2)

                # ── 5. convergence check ──────────────────────────────
                if k > 0 and k % 20 == 0:
                    torch.sub(u, prev_u, out=buf)
                    rel = buf.norm() / (prev_u.norm() + eps)
                    if rel < tol:
                        if verbose:
                            print('\nConverged at iteration {}.'.format(k + 1))
                        break
                    prev_u.copy_(u)

        if verbose:
            print('')
        return u.cpu()

    # ── finite-difference operators ───────────────────────────────────────
    #
    #  Convention:  dim 0 = batch,  dim 1 = row,  dim 2 = col.
    #  Forward diff with Neumann BC  →  last element = 0.
    #  Adjoint (= neg. backward diff):
    #       (Dᵀp)[0]   = −p[0]
    #       (Dᵀp)[i]   =  p[i−1] − p[i]
    #       (Dᵀp)[n−1] =  p[n−2]

    @staticmethod
    def _fwd(x, dim):
        """Forward difference along *dim* (Neumann BC)."""
        return x.diff(dim=dim, append=x.narrow(dim, x.size(dim) - 1, 1))

    @staticmethod
    def _dir_diff(x, mode):
        """Directional forward difference  D_θ(x)."""
        out = torch.zeros_like(x)
        if   mode == 0: out[:, :-1, :]   = x[:, 1:, :]   - x[:, :-1, :]    # vertical
        elif mode == 1: out[:, :-2, :-1] = x[:, 2:, 1:]  - x[:, :-2, :-1]  # 26.6° L
        elif mode == 2: out[:, :-1, :-1] = x[:, 1:, 1:]  - x[:, :-1, :-1]  # 45°   L
        elif mode == 3: out[:, :-2, 1:]  = x[:, 2:, :-1] - x[:, :-2, 1:]   # 26.6° R
        elif mode == 4: out[:, :-1, 1:]  = x[:, 1:, :-1] - x[:, :-1, 1:]   # 45°   R
        return out

    # ── adjoint operators  (target −= α · Dᵀ q) ─────────────────────────

    @staticmethod
    def _adj_1d(target, p, dim, a):
        """Adjoint of 1-D forward diff along *dim*, applied as target −= a·Dᵀp."""
        s = [slice(None)] * 3
        # first element
        s[dim] = 0
        target[tuple(s)].add_(p[tuple(s)], alpha=a)
        # interior
        s[dim] = slice(1, -1)
        s2 = list(s); s2[dim] = slice(None, -2)
        target[tuple(s)].sub_(p[tuple(s2)], alpha=a).add_(p[tuple(s)], alpha=a)
        # last element
        s[dim] = -1
        s2 = list(s); s2[dim] = -2
        target[tuple(s)].sub_(p[tuple(s2)], alpha=a)

    @classmethod
    def _adj_grad(cls, target, ph, pv, a):
        """target −= a · (∂ₕᵀ ph + ∂ᵥᵀ pv)."""
        cls._adj_1d(target, ph, 1, a)
        cls._adj_1d(target, pv, 2, a)

    @staticmethod
    def _adj_dir(target, q, mode, a):
        """target −= a · D_θᵀ q.

        For D(x)[i,j] = x[i+dr, j+dc] - x[i,j]  (active region only),
        the adjoint is:
            (Dᵀq)[p,q] = q[p-dr, q-dc]  (shifted)  −  q[p, q]  (in-place)
        each restricted to valid indices.

        target −= a·Dᵀq  ⟹  sub shifted term, add in-place term.
        """
        if mode == 0:    # vertical: dr=1, dc=0
            target[:, 1:, :].sub_(q[:, :-1, :], alpha=a)
            target[:, :-1, :].add_(q[:, :-1, :], alpha=a)
        elif mode == 1:  # 26.6° L: dr=2, dc=+1
            target[:, 2:, 1:].sub_(q[:, :-2, :-1], alpha=a)
            target[:, :-2, :-1].add_(q[:, :-2, :-1], alpha=a)
        elif mode == 2:  # 45° L:  dr=1, dc=+1
            target[:, 1:, 1:].sub_(q[:, :-1, :-1], alpha=a)
            target[:, :-1, :-1].add_(q[:, :-1, :-1], alpha=a)
        elif mode == 3:  # 26.6° R: dr=2, dc=−1
            target[:, 2:, :-1].sub_(q[:, :-2, 1:], alpha=a)
            target[:, :-2, 1:].add_(q[:, :-2, 1:], alpha=a)
        elif mode == 4:  # 45° R:  dr=1, dc=−1
            target[:, 1:, :-1].sub_(q[:, :-1, 1:], alpha=a)
            target[:, :-1, 1:].add_(q[:, :-1, 1:], alpha=a)

    # ── helpers ───────────────────────────────────────────────────────────

    def _to_tensor(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        return x.to(dtype=torch.float32)

    @staticmethod
    def _zeros2(ref):
        """Return a pair of zero tensors with the same shape/device as *ref*."""
        z = torch.zeros_like(ref)
        return z, z.clone()

    @staticmethod
    def _reflect_pad(t, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0):
        """Reflect-pad a 2-D tensor (torch requires 3-D+ for 4-element pad)."""
        if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
            return t
        return TF.pad(t.unsqueeze(0),
                      (pad_left, pad_right, pad_top, pad_bottom),
                      mode='reflect').squeeze(0)

    @staticmethod
    def _cosine_window(h, w, ov):
        """2-D cosine blending window: 1 in centre, 0 → 1 ramp over *ov* px."""
        win = torch.ones(h, w)
        if ov > 0:
            ramp = 0.5 * (1.0 - torch.cos(torch.linspace(0, math.pi, ov)))
            win[:ov, :]  *= ramp[:, None]
            win[-ov:, :] *= ramp.flip(0)[:, None]
            win[:, :ov]  *= ramp[None, :]
            win[:, -ov:] *= ramp.flip(0)[None, :]
        return win