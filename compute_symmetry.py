from sage.all import *

# Find the symmetry group of the decomposition mss, which is a list of triples
# [a_i, b_i, c_i] of matrices with compatible dimensions.

# this returns the group preserving a large set of invariants, and thus will
# certainly contain, and many times be equal to, the true group
def approximate_symmetry_group(mss):
    vertices = 6*len(mss)
    vertex_colors = [ list(range(6*len(mss))) ]
    edges = [ (6*i+j, 6*i+(j+1)%6) for i in range(len(mss)) for j in range(6) ] 
    edges.extend([ (6*i+(j+1)%6, 6*i+j) for i in range(len(mss)) for j in range(6) ] )

    # local factors transform together
    vertex_colors.append(list(range(vertices,vertices+6)))
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

    # rank is an invariant
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

    # geometric coincidences between the flats defined by decomposition elements are invariants
    # specifically dimensions of spans of unions must be preserved, which we encode here
    def flat_relations(flats,maxsize=2):
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

