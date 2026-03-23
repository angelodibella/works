# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
# See https://github.com/angelodibella/works for details.
from dataclasses import dataclass
from collections.abc import Sequence
import numpy as np
from .gf2 import logical_basis_css, row_basis_mod2

Shift = tuple[int, int]

def monomial_label(a:int,b:int)->str:
    if a==0 and b==0:
        return '1'
    out=[]
    if a:
        out.append('x' if a==1 else f'x^{a}')
    if b:
        out.append('y' if b==1 else f'y^{b}')
    return ''.join(out)

@dataclass(frozen=True)
class BBCodeSpec:
    l:int
    m:int
    A_terms: tuple[Shift, Shift, Shift]
    B_terms: tuple[Shift, Shift, Shift]
    name:str='BB'
    # Canonical benchmark metadata (from literature, not computed).
    # These override naive basis-weight computation when present.
    benchmark_n: int | None = None
    benchmark_k: int | None = None
    benchmark_d: int | None = None
    @property
    def half(self)->int:
        return self.l*self.m
    @property
    def n_data(self)->int:
        return 2*self.half
    @property
    def n_total(self)->int:
        return 4*self.half
    def idx(self,a:int,b:int)->int:
        return (a%self.l)*self.m + (b%self.m)
    def ab(self, idx:int)->tuple[int,int]:
        return (idx//self.m, idx%self.m)
    def term_apply(self, idx:int, shift:Shift)->int:
        a,b=self.ab(idx); da,db=shift
        return self.idx(a+da,b+db)
    def term_apply_T(self, idx:int, shift:Shift)->int:
        a,b=self.ab(idx); da,db=shift
        return self.idx(a-da,b-db)
    def monomial(self, idx:int)->str:
        a,b=self.ab(idx)
        return monomial_label(a,b)
    def permutation_matrix(self, shift:Shift)->np.ndarray:
        n=self.half
        mat=np.zeros((n,n),dtype=np.uint8)
        for i in range(n):
            mat[i,self.term_apply(i,shift)]=1
        return mat
    def polynomial_matrix(self, terms:Sequence[Shift])->np.ndarray:
        out=np.zeros((self.half,self.half),dtype=np.uint8)
        for t in terms:
            out ^= self.permutation_matrix(t)
        return out
    def hx(self)->np.ndarray:
        A=self.polynomial_matrix(self.A_terms)
        B=self.polynomial_matrix(self.B_terms)
        return np.concatenate([A,B],axis=1).astype(np.uint8)
    def hz(self)->np.ndarray:
        A=self.polynomial_matrix(self.A_terms)
        B=self.polynomial_matrix(self.B_terms)
        return np.concatenate([B.T,A.T],axis=1).astype(np.uint8)
    def row_basis_hx(self)->np.ndarray:
        return row_basis_mod2(self.hx())
    def row_basis_hz(self)->np.ndarray:
        return row_basis_mod2(self.hz())
    def logical_bases(self):
        return logical_basis_css(self.hx(), self.hz())
    def mapped_target_index(self, source_idx: int, term: Shift, transpose: bool, target_reg: str) -> int:
        if not transpose:
            return self.term_apply(source_idx, term)
        if target_reg in {'X', 'Z'}:
            return self.term_apply(source_idx, term)
        return self.term_apply_T(source_idx, term)

    def check_supports_x(self) -> list[list[int]]:
        out=[]
        for i in range(self.half):
            supp=[self.term_apply(i,t) for t in self.A_terms]
            supp += [self.half+self.term_apply(i,t) for t in self.B_terms]
            out.append(supp)
        return out
    def check_supports_z(self) -> list[list[int]]:
        out=[]
        for i in range(self.half):
            supp=[self.term_apply_T(i,t) for t in self.B_terms]
            supp += [self.half+self.term_apply_T(i,t) for t in self.A_terms]
            out.append(supp)
        return out

def build_bb72()->BBCodeSpec:
    return BBCodeSpec(6,6,((3,0),(0,1),(0,2)),((0,3),(1,0),(2,0)),'BB72',
                      benchmark_n=72, benchmark_k=12, benchmark_d=6)

def build_bb90()->BBCodeSpec:
    """[[90,8,10]] BB code on Z_15 x Z_3. A=x^9+y+y^2, B=1+x^2+x^7."""
    return BBCodeSpec(15,3,((9,0),(0,1),(0,2)),((0,0),(2,0),(7,0)),'BB90',
                      benchmark_n=90, benchmark_k=8, benchmark_d=10)

def build_bb108()->BBCodeSpec:
    """[[108,8,12]] BB code on Z_9 x Z_6. A=x^3+y+y^2, B=y^3+x+x^2.

    Uses the same polynomial pair as BB72/BB144 on a different lattice.
    This gives d=12 (verified by exhaustive 2^k search), not d=10 as in
    the canonical Bravyi et al. benchmark which uses different polynomials.
    """
    return BBCodeSpec(9,6,((3,0),(0,1),(0,2)),((0,3),(1,0),(2,0)),'BB108',
                      benchmark_n=108, benchmark_k=8, benchmark_d=12)

def build_bb144()->BBCodeSpec:
    return BBCodeSpec(12,6,((3,0),(0,1),(0,2)),((0,3),(1,0),(2,0)),'BB144',
                      benchmark_n=144, benchmark_k=12, benchmark_d=12)
