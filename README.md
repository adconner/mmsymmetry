Matrix multiplication rank decomposition symmetry
=

This is an implementation of 

1. computing the symmetry group of a given rank decomposition of the matrix
multiplication tensor, and

2. conversion of a symmetry group and orbit structure guess into the
corresponding jax search.

The matrix multiplication tensor $M_{uvw}$ lives in $(W \otimes U^*) \otimes (U
\otimes V^*) \otimes (V\otimes W^*)$ and has symmetry group contained in
$PGL(U)\times PGL(V)\times PGL(W) \rtimes S_3$. The element $g_1,g_2,g_3 \in
PGL(U)\times PGL(V)\times PGL(W)$ sends a rank 1 $(m_1,m_2,m_3) \mapsto
(g_3m_1g_1^{-1}, g_1m_2 g_2^{-1}, g_2 m_3 g_3^{-1})$, the cyclic permutation in
$(1,2,3) \in S_3$ act by $(m_1,m_2,m_3) \mapsto (m_3,m_1,m_2)$, and a
transposition $(1,2) \in S_3$ acts by $(m_1,m_2,m_3) \mapsto (m_2^t, m_1^t,
m_3^t)$. The factor permutation symmetries are only possible when the moved
factors have equal sizes. An element of this group is represented here by a
triple $(g_1,g_2,g_3)$ and a factor permutation in S_3, where the action of the
triple happens before the factor permutation.

Applying a symmetry of $M_{uvw}$ to a rank decomposition yields a possibly new
decomposition. In the special case that a decomposition in hand is preserved by
some symmetry (the terms being permuted), we say it is a symmetry of the
decomposition. The set of symmetries of a decomposition forms a finite group.
The function `symmetry_group` determines the symmetry group under some mild
assumptions on the $m_i$'s.

Given a finite subgroup of $g \le PGL(U) \times PGL(V) \times PGL(W) \rtimes
S_3$, it is always possible to find a finite group $G \le GL(U) \times GL(V) \times
GL(W) \rtimes S_3$ projecting to it (it is possible that $G$ may only be chosen
with a nontrivial kernel in the projection). We say that $G$ is a linearization
of $g$. Linearizations and their existence is convenient because it means in
the no transpose situation, for instance, that the action of $G$ is described
by three linear representations, rather than projective representations. We can
then use character theory to be quite systematic in studying linear
representations. The function `linearize_representation` finds a linear
representation projecting to a projective representation under some mild
assumptions.

Given a projective or linear representation, we may describe the orbit
structure of the rank 1's under the action. The triples in the decomposition
are collected into orbits; we pick one element $(m_1,m_2,m_3)$ from each orbit
and record its stabilizer $h$, the subgroup fixing the rank 1 tensor
corresponding to the triple. Consider the case that $g$ acts linearly. If $e\in
h$, then $e \cdot (m_1,m_2,m_3) = (\lambda_1 m_1, \lambda_2 m_2, \lambda_3
m_3)$, where $\lambda_1 \lambda_2 \lambda_3 = 1$. Then if there is no transpose
action, $\lambda_1,\lambda_2,\lambda_3$ define linear characters on $h$, and
$h$ with these three characters tensoring to the trivial character describe the
action on the orbit. More generally, if there is transpose symmetry, to
describe the action of the stabilizer, we consider that for $\mu_1\mu_2\mu_3 =
1$, $e \cdot (\mu_1 m_1, \mu_2 m_2, \mu_3 m_3) = (\lambda m_1, \lambda m_2,
\lambda m_3)$, and the action can be viewed as the assignment
$(\mu_1,\mu_2,\mu_3) \mapsto (\lambda_1,\lambda_2,\lambda_3)$ via some element
of $(\mathbb{C}^*)^{\times 3} \rtimes S_3$. More generally, therefore, we
record the stabilizer $h$ and the corresponding homomorphism $h \to
(\mathbb{C}^*)^{\times 3} \rtimes S_3$ to record the action of the stabilizer.
If $g$ is a projective, rather than linear representation, we no longer have
linear characters or more generally homomorphisms into the wreath product, but
we can record the way distinguished linear lifts of a generating set act. The
function `orbit_structure` computes this information given a decomposition and
its symmetry group.

Finally, given a group with either projective or linear representation, and
given an orbit structure, `get_map_to_rank1s` computes a function applicable to
numpy or jax vectors which parameterizes decompositions with the corresponding
symmetry and orbit structure. More precisely, the set of all the matrix entries
$\{ (m_1^i,m_2^i,m_3^i) \mid i \}$ with prescribed group and orbit structure
forms a linear subspace, and and `get_map_to_rank1s` picks a computable
parameterization of this subspace.

Caveat: If we search for decompositions in the same way as before, fixing in
advance an allowable sublattice the parameters may take, then our choice of
parameterization affects the decompositions we consider. In other words there
is no longer any privileged basis for the symmetric rank 1 matrix subspace with
respect to which to base our lattice. To take this approach of searching in a
lattice rather than a vector space in a principled way, I expect we need to
allow the lattice to be determined by the agent.

For simple examples of corresponding jax search, see `search_2x2.py`,
`search_4x4.py`, `search_2x2_notrans.py`, and `search_4x4_notrans.py`, where we
take the symmetry group of Strassen and the AlphaEvolve found 48 and look for
decompositions with only this symmetry. The notrans versions pass to only the
$GL(U) x GL(V) x GL(W)$ part of the symmetry and describe the linear
representations only by their characters, reconstructing an isomorphic
representation in which to work on the fly.

The code makes heavy use of gap for the group theory computations through the
SageMath interface. Nix is used to ensure reproducibility of the environment.
Of course one can set up an environment manually following `shell.nix` if
desired. Due to the way SageMath is currently packaged, one hase to use the
sage included python.

Steps to run: 

1. Install nix (https://nixos.org/download/)

2. Enter environment
```
$ nix-shell
```
  or, if you prefer, use direnv

3.
```
# to compute symmetry and orbit structure for AlphaEvolve found decompositions
$ sage -python ae_decomps.py 

# to run a simple jax search for a 4x4 rank 48 decomposition with the same 
# symmetry as the AlphaEvolve found one
$ sage -python search_4x4.py

# similarly
$ sage -python search_4x4_notrans.py
$ sage -python search_2x2.py
$ sage -python search_2x2_notrans.py
```

