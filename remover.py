import math
import torch
import torch.nn.functional as TF

from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union


_N = 5
_N_VARS = 1 + _N


class UniversalStripeRemover:

    def __init__(
        self,
        mu1: float = 0.33,
        mu2: float = 0.003,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mu1 = mu1
        self.mu2 = mu2
        self.tau = 0.35
        self.sigma = 0.35

    def process(
        self,
        image: Union[torch.Tensor, Any],
        iterations: int = 500,
        tol: float = 1e-5,
        proj: bool = True,
        verbose: bool = True,
    ) -> torch.Tensor:
        x = self._to_tensor(x=image)
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)

        result = self._solve(
            data = x,
            iterations = iterations,
            tol = tol,
            proj = proj,
            verbose = verbose,
        )

        return result.squeeze(0) if squeeze else result

    def process_tiled(
        self,
        image: Union[torch.Tensor, Any],
        n: int = 1,
        iterations: int = 500,
        tol: float = 1e-5,
        overlap: int = 64,
        proj: bool = True,
        verbose: bool = True,
    ) -> torch.Tensor:
        data = self._to_tensor(x=image)
        if data.dim() == 3:
            data = data.squeeze(0)

        if n <= 1:
            return self.process(
                image = data,
                iterations = iterations,
                tol = tol,
                proj = proj,
                verbose = verbose,
            )

        h, w = data.shape

        data = self._reflect_pad(
            t = data,
            pad_bottom = (n - h % n) % n,
            pad_right = (n - w % n) % n,
        )
        hp, wp = data.shape
        th_core, tw_core = hp // n, wp // n

        ov = min(overlap, th_core // 4, tw_core // 4)
        ov = max(ov, 0)

        data = self._reflect_pad(
            t = data,
            pad_top = ov,
            pad_bottom = ov,
            pad_left = ov,
            pad_right = ov,
        )

        th, tw = th_core + 2 * ov, tw_core + 2 * ov

        tiles = [
            data[i * th_core : i * th_core + th, j * tw_core : j * tw_core + tw]
            for i in range(n)
            for j in range(n)
        ]
        batch = torch.stack(tensors=tiles)
        if verbose:
            print(f"Tiling {n}x{n}: {n * n} tiles of {th}x{tw}, overlap={ov}")

        results = self.process(
            image = batch,
            iterations = iterations,
            tol = tol,
            proj = proj,
            verbose = verbose,
        )

        weight = self._cosine_window(h=th, w=tw, ov=ov)
        canvas = torch.zeros(hp + 2 * ov, wp + 2 * ov)
        wsum = torch.zeros_like(input=canvas)

        for idx, (i, j) in enumerate((i, j) for i in range(n) for j in range(n)):
            y, x = i * th_core, j * tw_core
            canvas[y : y + th, x : x + tw] += results[idx] * weight
            wsum[y : y + th, x : x + tw] += weight

        canvas /= wsum.clamp(min=1e-9)
        return canvas[ov : ov + hp, ov : ov + wp][:h, :w]

    def _solve(
        self,
        data: torch.Tensor,
        iterations: int,
        tol: float,
        proj: bool,
        verbose: bool,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=torch.float32)
        ts = self.tau * self.sigma
        lam = self.mu1 / self.sigma
        qc = 1.0 / self.sigma
        rc = self.mu2 / self.sigma
        eps = 1e-9

        u = data.clone()
        s = [torch.zeros_like(input=data) for _ in range(_N)]

        ph, ph_ = self._zeros2(ref=data)
        pv, pv_ = self._zeros2(ref=data)

        q, q_ = zip(*(self._zeros2(ref=data) for _ in range(_N)))
        r, r_ = zip(*(self._zeros2(ref=data) for _ in range(_N)))
        q_list = list(q)
        q_ext = list(q_)
        r_list = list(r)
        r_ext = list(r_)

        prev_u = u.clone()
        buf = torch.empty_like(input=data)

        with torch.no_grad():
            for k in range(iterations):
                if verbose:
                    print(f"\rIteration: {k + 1} / {iterations}", end="")

                self._adj_grad(target=u, ph=ph_, pv=pv_, a=ts)
                for i in range(_N):
                    self._adj_dir(target=s[i], q=q_ext[i], mode=i, a=ts)
                    s[i].sub_(r_ext[i], alpha=ts)

                buf.copy_(data)
                for si in s:
                    buf.sub_(si)
                buf.sub_(u).div_(_N_VARS)
                u.add_(buf)
                for si in s:
                    si.add_(buf)

                if proj:
                    torch.clamp(input=u, max=0, out=buf)
                    buf.add_((u - 1).clamp_(min=0))
                    buf.div_(_N)
                    for si in s:
                        si.add_(buf)
                    u.clamp_(min=0, max=1)

                ph_.copy_(ph)
                pv_.copy_(pv)

                zh = ph + self._fwd(x=u, dim=1)
                zv = pv + self._fwd(x=u, dim=2)
                norm = torch.sqrt(input=zh * zh + zv * zv).clamp_(min=eps)
                scale = (lam / norm).clamp_(max=1.0)
                ph = zh.mul_(scale)
                pv = zv.mul_(scale)
                ph_.mul_(-1).add_(ph, alpha=2)
                pv_.mul_(-1).add_(pv, alpha=2)

                for i in range(_N):
                    q_ext[i].copy_(q_list[i])
                    q_list[i] = (q_list[i] + self._dir_diff(x=s[i], mode=i)).clamp_(min=-qc, max=qc)
                    q_ext[i].mul_(-1).add_(q_list[i], alpha=2)

                    r_ext[i].copy_(r_list[i])
                    r_list[i] = (r_list[i] + s[i]).clamp_(min=-rc, max=rc)
                    r_ext[i].mul_(-1).add_(r_list[i], alpha=2)

                if k > 0 and k % 20 == 0:
                    torch.sub(input=u, other=prev_u, out=buf)
                    rel = buf.norm() / (prev_u.norm() + eps)
                    if rel < tol:
                        if verbose:
                            print(f"\nConverged at iteration {k + 1}.")
                        break
                    prev_u.copy_(u)

        if verbose:
            print("")
        return u.cpu()

    @staticmethod
    def _fwd(
        x: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        return x.diff(dim=dim, append=x.narrow(dim=dim, start=x.size(dim) - 1, length=1))

    @staticmethod
    def _dir_diff(
        x: torch.Tensor,
        mode: int,
    ) -> torch.Tensor:
        out = torch.zeros_like(input=x)
        if mode == 0:
            out[:, :-1, :] = x[:, 1:, :] - x[:, :-1, :]
        elif mode == 1:
            out[:, :-2, :-1] = x[:, 2:, 1:] - x[:, :-2, :-1]
        elif mode == 2:
            out[:, :-1, :-1] = x[:, 1:, 1:] - x[:, :-1, :-1]
        elif mode == 3:
            out[:, :-2, 1:] = x[:, 2:, :-1] - x[:, :-2, 1:]
        elif mode == 4:
            out[:, :-1, 1:] = x[:, 1:, :-1] - x[:, :-1, 1:]
        return out

    @staticmethod
    def _adj_1d(
        target: torch.Tensor,
        p: torch.Tensor,
        dim: int,
        a: float,
    ) -> None:
        s = [slice(None)] * 3
        s[dim] = 0
        target[tuple(s)].add_(p[tuple(s)], alpha=a)

        s[dim] = slice(1, -1)
        s2 = list(s)
        s2[dim] = slice(None, -2)
        target[tuple(s)].sub_(p[tuple(s2)], alpha=a).add_(p[tuple(s)], alpha=a)

        s[dim] = -1
        s2 = list(s)
        s2[dim] = -2
        target[tuple(s)].sub_(p[tuple(s2)], alpha=a)

    @classmethod
    def _adj_grad(
        cls,
        target: torch.Tensor,
        ph: torch.Tensor,
        pv: torch.Tensor,
        a: float,
    ) -> None:
        cls._adj_1d(target=target, p=ph, dim=1, a=a)
        cls._adj_1d(target=target, p=pv, dim=2, a=a)

    @staticmethod
    def _adj_dir(
        target: torch.Tensor,
        q: torch.Tensor,
        mode: int,
        a: float,
    ) -> None:
        if mode == 0:
            target[:, 1:, :].sub_(q[:, :-1, :], alpha=a)
            target[:, :-1, :].add_(q[:, :-1, :], alpha=a)
        elif mode == 1:
            target[:, 2:, 1:].sub_(q[:, :-2, :-1], alpha=a)
            target[:, :-2, :-1].add_(q[:, :-2, :-1], alpha=a)
        elif mode == 2:
            target[:, 1:, 1:].sub_(q[:, :-1, :-1], alpha=a)
            target[:, :-1, :-1].add_(q[:, :-1, :-1], alpha=a)
        elif mode == 3:
            target[:, 2:, :-1].sub_(q[:, :-2, 1:], alpha=a)
            target[:, :-2, 1:].add_(q[:, :-2, 1:], alpha=a)
        elif mode == 4:
            target[:, 1:, :-1].sub_(q[:, :-1, 1:], alpha=a)
            target[:, :-1, 1:].add_(q[:, :-1, 1:], alpha=a)

    def _to_tensor(
        self,
        x: Any,
    ) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(data=x)
        return x.to(dtype=torch.float32)

    @staticmethod
    def _zeros2(
        ref: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros_like(input=ref)
        return z, z.clone()

    @staticmethod
    def _reflect_pad(
        t: torch.Tensor,
        pad_top: int = 0,
        pad_bottom: int = 0,
        pad_left: int = 0,
        pad_right: int = 0,
    ) -> torch.Tensor:
        if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
            return t

        return TF.pad(
            input = t.unsqueeze(0),
            pad = (pad_left, pad_right, pad_top, pad_bottom),
            mode = "reflect",
        ).squeeze(0)

    @staticmethod
    def _cosine_window(
        h: int,
        w: int,
        ov: int,
    ) -> torch.Tensor:
        win = torch.ones(h, w)
        if ov > 0:
            ramp = 0.5 * (1.0 - torch.cos(input=torch.linspace(start=0, end=math.pi, steps=ov)))
            win[:ov, :] *= ramp[:, None]
            win[-ov:, :] *= ramp.flip(dims=(0,))[:, None]
            win[:, :ov] *= ramp[None, :]
            win[:, -ov:] *= ramp.flip(dims=(0,))[None, :]
        return win