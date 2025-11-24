from sage.all import gap,matrix,UniversalCyclotomicField,block_matrix,identity_matrix,sqrt
import numpy as np
from itertools import product

def get_map_to_rank1s(rep,orbit_structure):
    g = rep['g']
    tripf = rep['tripf']
    fac_perm_map = rep['fac_perm_map']
    m,n,l = [int(m[1].Length()) for m in tripf(g.Identity())]
    def action_on_trip(e,M=None):
        u,v,w = [matrix(UniversalCyclotomicField(),m) for m in tripf(e)]
        factor_actions = ( w.T.tensor_product(~u), u.T.tensor_product(~v), v.T.tensor_product(~w) )
        fac_perm = e ** fac_perm_map
        if M is None:
            M = gap.PermutationMat(fac_perm,3).sage()
        blocks = [[ act / e if e != 0 else 0 for e in r ] for r, act in zip(M,factor_actions) ]
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
        fac_mults = fac_mults.sage()
        eqs = []
        for e, M in zip(h.GeneratorsOfGroup(), fac_mults):
            act = -action_on_trip(e,M)
            act += identity_matrix(act.nrows())
            eqs.append(act)
        
        if len(eqs) > 0:
            eqs = block_matrix([eqs])
            embed1 = eqs.left_kernel_matrix() # can pick favorite basis here
        else:
            embed1 = identity_matrix(UniversalCyclotomicField(), l*m+m*n+n*l)
        
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

    def param_to_trips(x,np=np):
        res = np.vstack([ np.einsum('ijk,i->jk', embed, x[xstart:xend]) for embed, (xstart, xend) in orbit_embeds ])
        return (res[:, :l*m].T, res[:, l*m:l*m+m*n].T, res[:, l*m+m*n:].T)
    
    return param_to_trips, params, orbit_embeds

def matrixmult(m,n,l):
    T=np.zeros((l*m,m*n,n*l))
    for i,j,k in product(range(m),range(n),range(l)):
        T[k*m+i, i*n+j, j*l+k] = 1
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
    
def get_gap_code(rep,orbit_structure):
    m,n,l = rep['tripf'](rep['g'].Identity()).List(gap.Length).sage()
    return f'''g := Group([{','.join([str(e) for e in rep['g'].GeneratorsOfGroup()])}]);
acts := [ {','.join([ f'\n  [ [ {',\n    '.join([f'{m}' for m in rep['tripf'](e)])} ], {e**rep['fac_perm_map']}]' for e in rep['g'].GeneratorsOfGroup()])} ];
fac_perm_map := GroupHomomorphismByImages(g, Group(List(acts, p->p[2])));
acts := List(acts, p -> TripToMat(p[1], p[2]));
hom := GroupHomomorphismByImages(g,Group(acts));
tripf := e -> MatToTrip(e ^ hom, [{m},{n},{l}])[1];
orbits := [ {','.join([ f'\n  [ {h},\n  [ { ',\n    '.join([f'{m}' for m in ms])} ] ]' for h, ms in gap(orbit_structure)])} ];
'''

# If rep has no transpose action, then we can refer to everything in sight as
# indices into gaps computed lists for the objects. This is to indicate that we
# can make gap do most of the work for us. The only control we give up is in
# the particular bases gap chooses to represent the action in, which has
# implications if we insist to look inside a distinguished lattice (integers or
# half gauss integers, etc) of solutions.
def get_gap_code_simple(rep,orbit_structure):
    m,n,l = rep['tripf'](rep['g'].Identity()).List(gap.Length).sage()
    assert rep['fac_perm_map'].Image().Size().sage() == 1
    group_id = rep['g'].IdSmallGroup()
    gstd = gap.SmallGroup(group_id)
    iso = gstd.IsomorphismGroups(rep['g'])
    chis = gstd.ConjugacyClasses().List(f'e -> List({rep["tripf"].name()}(Representative(e)^{iso.name()}),TraceMat)')
    chis = gap(f'List([1,2,3], i -> Character({gstd.name()},List({chis.name()}, vals -> vals[i])))')
    irr = gstd.Irr().Set()
    chis = gap(f'List({chis.name()}, chi -> Concatenation(List(ConstituentsOfCharacter(chi), psi -> ListWithIdenticalEntries(ScalarProduct(chi,psi), Position({irr.name()},psi)))))')
    chis = chis.List(gap.Set)
    subgroup_classes = gstd.ConjugacyClassesSubgroups()
    os = []
    for h, ms in orbit_structure:
        homs = gap(f'List([1,2], i -> GroupHomomorphismByImages({h.name()}, Group(List({ms.name()}, m -> [[m[i][i]]]),[[1]])))')
        h = iso.PreImages(h)
        hi = subgroup_classes.PositionProperty(f'cl -> {h.name()} in cl')
        hstd = subgroup_classes[hi].CanonicalRepresentativeOfExternalSet()
        e = gap.RepresentativeAction(gstd, hstd, h)
        conjiso = gap.ConjugatorIsomorphism(hstd, e)
        psis = gap(f'List({homs.name()}, hom -> Character({hstd.name()}, List(ConjugacyClasses({hstd.name()}), cl -> (((Representative(cl)^{conjiso.name()})^{iso.name()})^hom)[1][1] )))')
        lin_irr = hstd.LinearCharacters().Set()
        psis = psis.List(f'psi->Position({lin_irr.name()},psi)')
        os.append([hi,psis])
    return f'''g := {group_id};
chis := {chis};
orbits := {os};
g := SmallGroup(g);
irr := Set(Irr(g));
reps := List(chis, function(chi)
    local reps, imgs;
    reps := List(chi, i -> IrreducibleRepresentationsDixon(g, irr[i]) );
    imgs := List(GeneratorsOfGroup(g), e-> DirectSumMat(List(reps, rep -> e^rep)));
    return GroupHomomorphismByImages(g, Group(imgs));
end);
tripf := e -> List(reps, rep->e^rep);
subgroups := List(ConjugacyClassesSubgroups(g), CanonicalRepresentativeOfExternalSet);
orbits := List(orbits, function(p)
    local h, lin_irr, chis, Ms;
    h := subgroups[p[1]];
    lin_irr := Set(LinearCharacters(h));
    chis := List(p[2], i -> lin_irr[i]);
    Ms := List(GeneratorsOfGroup(h), e -> DiagonalMat([e^chis[1], e^chis[2], 1/(e^chis[1]*e^chis[2])]));
    return [h, Ms];
end);
fac_perm_map := GroupHomomorphismByImages(g, ListWithIdenticalEntries(Length(GeneratorsOfGroup(g)),()));
'''
