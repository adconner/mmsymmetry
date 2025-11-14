# Here we look to find the symmetry group of the decomposition mss, which is a
# list of triples [a_i, b_i, c_i] of matrices with compatible dimensions.
# Ie a_i in W ot U^*, b_i in U ot V^* and c_i in V ot W^*

# We wish to determine when a permutation of the triples is achievable under
# the group of symmetries of the matrix multiplication tensor, ie under the action 
# a,b,c -> g_3^-1 a g_1, g_1^-1 b g_2, g_2^-1 c, g_3, where g_1 in PGL(U), g_2
# in PGL(V), and g_3 in PGL(W). Actually, we allow something slightly
# more general, the above plus rescalings a,b,c -> lambda a, mu b, nu c, where
# lambda mu nu = 1. Additionally, if the dimensions are appropriate we allow
# cyclic symmetries a,b,c -> b,c,a and transpose symmetries a,b,c -> b^t,a^t,c^t.

# Our fundamental tool is to consider the projective configuration of flats in
# P(U) given by the a_i(W^*)^perp and b_i(V). Any PGL(U) occurring as a
# symmetry must preserve this set of flats. Likewise we consider the
# corresponding configurations in P(V) and P(W), which must also be preserved.

# There are two fundamental ideas here: 
# (1) find the permutation group preserving some geometric coincidences in
# these projective configurations to reduce to a much smaller group of the full
# symmetric group. This is done in approximate_symmetry_group below. 
# (2) to determine a "base" of decomposition elements with the property that
# the image of this small set determines the corrpesponding elements of
# PGL(U), PGL(V), and PGL(W). When we consider points in P^n, it is a classical
# fact that n+2 points in general position determine a unique element of PGL_n,
# and this element is efficiently computable. It seems like a more difficult
# problem and less well understood to determine projective equivalence of
# configurations of possibly larger flats. Luckily for us, it seems low rank
# decompositions of the matrix multiplication tensor typically contain many
# terms where the a_i, b_i, and c_i have rank 1, so we can use n+2 of these in
# general position to quickly determine a PGL equivalence if it exists. This is
# done in symmetry_group below

from sage.all import *
from itertools import groupby
import random

# Solve the problem where mss contains matrix triples of rank 1,1,1, with
# sufficiently many column spaces in general position
def symmetry_group(mss):
    # Find M.nrows() + 1 colmuns of M in general position.
    # This can fail to find a solution even if it exists but should succeed in
    # the common case. If we need to we can try harder here. 
    def general_frame_indices(M,tries=10):
        def try1():
            cols = list(range(M.ncols()))
            random.shuffle(cols)
            Me = M[:,cols].echelon_form()
            jxs = Me.pivots()
            try:
                jxs += (next(j for j in range(jxs[-1]+1,Me.ncols()) if all(e != 0 for e in Me[:,j])),)
                return tuple(sorted([cols[j] for j in jxs]))
            except StopIteration:
                raise ValueError("general_frame_indices: Did not find general frame")
        for t in range(tries-1):
            try:
                return try1()
            except ValueError:
                pass
        return try1()
    row = lambda m: next(r for r in m.rows() if not r.is_zero())
    column = lambda m: row(m.T)
    mss111 = [ ms for ms in mss if all(m.rank() == 1 for m in ms) ]

    us = [list(v) for a,b,c in mss111 for v in [row(a), column(a), row(b), column(b), row(c), column(c)]]
    g = approximate_symmetry_group(mss111,2)
    print(f'Containing symmetry group has order {g.Order()}')
    Ms = [matrix([column(ms[(i+1)%3]) for ms in mss111]).T for i in range(3)]
    frame_ixs = [[6*i+2*((f+1)%3)+1+1 for i in general_frame_indices(M)] for f,M in enumerate(Ms)]
    gap.Read('"compute_symmetry.g"');
    g, tripForPerm = gap.SymmetryGroupUsingPoints(us, g, frame_ixs, mss)
    return (g,tripForPerm)

# View a ot b ot c in A ot B ot C as living in A op B op C / C^2 with our
# decomposition terms having distinguished lifts. Similarly, we have an
# action defined in GL(A) x GL(B) x GL(C) / C^2, and we have distinguished
# lifts by asking that the first nonzero entry is 1 in the corresponding
# elements of GL(U), GL(V), GL(W). Note that we make no effort to pick
# distinguished lifts with any nice property (such as forming a subgroup), as
# these are not guaranteed to exist. For each orbit, we pick a point a op b op
# c, record its stabilizer h, and record for each element e in h the element of
# C^2 required to bring e . (a op b op c) back to a op b op c. Observe that
# this data is constant along conjugacy classes of h, so we record this
# information for each class. So, for each orbit we give h and 3 class
# functions of h multiplying to 1, corresponding to the post action in each
# tensor factor required to return to the distinguished lift.

# If our projective representation is actually a linear representations (ie if
# we can choose distinguished lifts of g forming a subgroup), the three class
# functions are linear characters. 
def orbit_structure(g,tripForPerm,mss):
    gap.Read('"compute_symmetry.g"');
    return gap.OrbitStructure(g,tripForPerm,mss)
    
# Compute a group preserving a large set of invariants, and thus one which will
# certainly contain, and many times be equal to, the true group. The group is
# computed as the group of automorphisms of a graph which encodes the
# invariants. One can increase maxsize to determine and exploit geometric
# coincidences involving at most maxsize elements of the flats.
def approximate_symmetry_group(mss,maxsize=2):
    vertices = 6*len(mss)
    vertex_colors = [ list(range(6*len(mss))) ]
    edges = [ (6*i+j, 6*i+(j+1)%6) for i in range(len(mss)) for j in range(6) ] 
    edges.extend([ (6*i+(j+1)%6, 6*i+j) for i in range(len(mss)) for j in range(6) ] )

    # local factors transform together, preserving the local sizes
    a,b,c = mss[0]
    local_sizes = [ a.nrows(), a.ncols(), b.nrows(), b.ncols(), c.nrows(), c.ncols() ]
    vertex_colors.extend([[vertices+j for j,_ in fs] for _, fs in \
                          groupby(sorted(enumerate(local_sizes),key = lambda p: p[1]), key = lambda p: p[1])])
    for i in range(len(mss)):
        for j in range(6):
            edges.append(( i*6+j, vertices+j ))
    vertices += 6
    
    # factors transform together
    vertex_colors.append(list(range(vertices,vertices+3)))
    for i in range(len(mss)):
        for j in range(3):
            edges.append(( i*6+2*j, vertices+j ))
            edges.append(( i*6+2*j+1, vertices+j ))
    vertices += 3

    # rank is invariant
    rank_rels_color = []
    rank_vertices = {}
    for i,ms in enumerate(mss):
        for f,m in enumerate(ms):
            if m.rank() not in rank_vertices:
                rank_vertices[m.rank()] = vertices
                rank_rels_color.append(vertices)
                vertices += 1
            edges.append((rank_vertices[m.rank()], 6*i+2*f))
            edges.append((rank_vertices[m.rank()], 6*i+2*f+1))
    vertex_colors.append(rank_rels_color)

    # geometric coincidences between the flats defined by decomposition elements are invariant,
    # specifically dimensions of spans of unions must be preserved, which we encode here
    def flat_relations(flats,maxsize=maxsize):
        flats = [(i,fl) for (i,fl) in enumerate(flats) if fl.nrows() > 0 ]
        rels_by_last = {}
        for relsize in range(2,maxsize+1):
            def dfs(ix):
                for i in range(0 if len(ix) == 0 else ix[-1]+1,len(flats)):
                    have_subrel = False
                    for rel in rels_by_last.get(i,[]):
                        if all(j in ix for j in rel):
                            have_subrel = True
                            break
                    if have_subrel:
                        continue
                    jx = ix+(i,)
                    M = block_matrix([[copy(flats[j][1])] for j in jx])
                    r = M.rank()
                    if len(jx) == relsize:
                        if r < min(M.dimensions()):
                            rels_by_last.setdefault(i,[]).append(ix)
                            yield (jx,r)
                    else:
                        for res in dfs(jx):
                            yield res
            for ix,r in dfs(()):
                yield ([flats[i][0] for i in ix],r)
    def ech(m):
        return m.echelon_form()[:m.rank()]
    flat_rels_colors = {}
    for f in range(3):
        flats = [ ech(ms[(f+1)%3].T) for msi,ms in enumerate(mss) ]
        flats += [ ms[f].right_kernel_matrix() for msi,ms in enumerate(mss) ]
        for ix,r in flat_relations(flats):
            # print('a',f,ix,r)
            edges.append((vertices,vertices+1))
            for i in ix:
                if i < len(mss):
                    edges.append((6*i+2*((f+1)%3), vertices))
                else:
                    i -= len(mss)
                    edges.append((6*i+2*((f+1)%3), vertices+1))
            flat_rels_colors.setdefault(r,[]).extend([vertices,vertices+1])
            vertices += 2
        flats = [ ech(ms[f]) for msi,ms in enumerate(mss) ]
        flats += [ ms[(f+1)%3].T.right_kernel_matrix() for msi,ms in enumerate(mss) ]
        for ix,r in flat_relations(flats):
            # print('b',f,ix,r)
            edges.append((vertices,vertices+1))
            for i in ix:
                if i < len(mss):
                    edges.append((6*i+2*f+1, vertices))
                else:
                    i -= len(mss)
                    edges.append((6*i+2*f+1, vertices+1))
            flat_rels_colors.setdefault(r,[]).extend([vertices,vertices+1])
            vertices += 2
    vertex_colors.extend(flat_rels_colors.values())

    gap.LoadPackage('"grape"');
    gap.Read('".enable_bliss.g"');
    g = gap.EdgeOrbitsGraph(gap('Group(())'), [[e+1,f+1] for e,f in edges], vertices)
    
    return gap.Action( gap.AutGroupGraph(g, [[i+1 for i in color] for color in vertex_colors]), list(range(1,6*len(mss)+1)) )
    # return gap.AutGroupGraph(g, [[i+1 for i in color] for color in vertex_colors])

