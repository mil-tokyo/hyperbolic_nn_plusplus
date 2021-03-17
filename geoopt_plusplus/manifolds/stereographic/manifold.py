from typing import List, Optional, Tuple

import torch.nn

import geoopt

from . import math


class PoincareBall(geoopt.PoincareBall):
    def weighted_midpoint(
        self,
        xs: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        *,
        reducedim: Optional[List[int]] = None,
        dim: int = -1,
        keepdim: bool = False,
        lincomb: bool = False,
        posweight=False,
        project=True,
    ):
        mid = math.weighted_midpoint(
            xs=xs,
            k=-self.c,
            weights=weights,
            reducedim=reducedim,
            dim=dim,
            keepdim=keepdim,
            lincomb=lincomb,
            posweight=posweight,
        )
        if project:
            return math.project(mid, k=-self.c, dim=dim)
        else:
            return mid

    def weighted_midpoint_bmm(
        self,
        xs: torch.Tensor,
        weights: torch.Tensor,
        lincomb: bool = False,
        project=True,
    ):
        mid = math.weighted_midpoint_bmm(
            xs=xs,
            weights=weights,
            k=-self.c,
            lincomb=lincomb,
        )
        if project:
            return math.project(mid, k=-self.c, dim=-1)
        else:
            return mid

    def mobius_coadd(
        self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = math.mobius_coadd(x, y, k=-self.c, dim=dim)
        if project:
            return math.project(res, k=-self.c, dim=dim)
        else:
            return res
    
    def dist_matmul(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return math.dist_matmul(x, y, k=-self.c)

    def dist2plane_matmul(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        p: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        return math._dist2plane_matmul(x, z, p, k=-self.c)


class PoincareBallExact(PoincareBall):
    reversible = True
    retr_transp = PoincareBall.expmap_transp
    transp_follow_retr = PoincareBall.transp_follow_expmap
    retr = PoincareBall.expmap

    def extra_repr(self):
        return "exact"
