from sage.all import gap,matrix,UniversalCyclotomicField
import numpy as np
from itertools import product

def get_map_to_rank1s(rep,orbit_structure):
    g = rep['g']
    tripf = rep['tripf']
    fac_perm_map = rep['fac_perm_map']
    def action_on_trip(e,M=None):
        u,v,w = [matrix(UniversalCyclotomicField(),m) for m in tripf(e)]
        factor_actions = ( w.T.tensor_product(~u), u.T.tensor_product(~v), v.T.tensor_product(~w) )
        fac_perm = e ** fac_perm_map
        if M is None:
            M = gap.PermutationMat(fac_perm,3).sage()
        blocks = [[ e * act for e in r ] for r, act in zip(M,factor_actions) ]
        if fac_perm.SignPerm() == -1:
            def trans(nn):
                n = sqrt(nn)
                return matrix([[1 if i1==j and j1==i else 0 for i1,j1 in product(range(n),range(n))] 
                               for i,j in product(range(n),range(n))])
            blocks = [[0 if b == 0 else b*trans(b.nrows()) for b in r] for r in blocks ] 
        return block_matrix(blocks)
    orbit_embeds = []
    params = 0
    for h, fac_mults in list(orbit_structure):
        eqs = []
        for e, M in zip(h.GeneratorsOfGroup(), fac_mults.sage()):
            act = -action_on_trip(e,M)
            act += identity_matrix(act.nrows())
            eqs.append(act)
        eqs = block_matrix([eqs])
        
        embed1 = eqs.left_kernel_matrix() # can pick favorite basis here
        
        embeds = []
        for e in g.RightCosets(h):
            e = e.Representative()
            embeds.append(embed1 * action_on_trip(e))
        embeds = np.array(embeds).transpose(1,0,2) # (input, orbitsize, triple)
        try:
            embeds = embeds.astype(np.double)
        except TypeError:
            embeds = embeds.astype(np.complex128)
        xbounds = (params, params+embed1.nrows())
        orbit_embeds.append((embeds,xbounds))
        params += embed1.nrows()

    m,n,l = [int(m[1].Length()) for m in tripf(g.Identity())]
    def param_to_trips(x,np=np):
        res = np.vstack([ np.einsum('ijk,i->jk', embed, x[xstart:xend]) for embed, (xstart, xend) in orbit_embeds ])
        return (res[:, :l*m], res[:, l*m:l*m+m*n], res[:, l*m+m*n:])
    
    return param_to_trips, params

def matrixmult(m,n,l):
    T=np.zeros((l*m,m*n,n*l))
    for i,j,k in product(range(m),range(n),range(l)):
        T[k*m+i, i*n+j, j*l+i] = 1
    return T

def evaluate(m,n,l, rep, orbit_structure, x, post = None):
    T = matrixmult(m,n,l)
    tripf, params = get_map_to_rank1s(rep, orbit_structure)
    assert x.shape == (params,)
    u,v,w = tripf(x)
    if post is not None:
        u,v,w = post(u,v,w)
    residual = np.einsum('ir,jr,kr->ijk',u,v,w) - matrixmult(m,n,l)
    return np.linalg.norm(residual)
    
