LoadPackage("gauss");

# Pick a PGL transformation normalizing the column vectors of M in some way, if it exists
ProjectiveNormalizationMap := function(M)
  local v,n,i,j,ifirst;
  v := Length(M);
  n := Length(M[1]);
  M := TransposedMat(Concatenation(TransposedMat(M), IdentityMat(v)));
  M := MutableCopyMat(EchelonMat(M).vectors);
  for i in [2..v] do
    for j in [1..n] do
      if IsZero(M[i][j]) then
        continue;
      fi;
      ifirst := First([1..v],i->not IsZero(M[i][j]));
      if ifirst >= i then
        continue;
      fi;
      M[i] := M[i] * M[ifirst,j] / M[i,j];
    od;
  od;
  return List(M, r->r{[n+1..n+v]});
end;

TripAction := function (gs,fac_perm,ms)
  ms := [gs[3]*ms[1]*Inverse(gs[1]), gs[1]*ms[2]*Inverse(gs[2]), gs[2]*ms[3]*Inverse(gs[3])];
  ms := Permuted(ms, fac_perm);
  if SignPerm(fac_perm) = -1 then
    ms := List(ms,TransposedMat);
  fi;
  return ms;
end;

SymmetryGroupUsingPoints := function(uss, g, mss)
  local local_fac_perm_map, term_perm_map111, to3fac, normalize,
        tripForPerm111, gens, mono, term_perm_map, fac_perm_map, tripf, det1Mult;

  local_fac_perm_map := ActionHomomorphism(g, List([1..6], j->List([0..Length(uss[1])-1], i->6*i+j)), OnSets);
  term_perm_map111 := ActionHomomorphism(g, List([0..Length(uss[1])-1], i->List([1..6], j->6*i+j)), OnSets);
  to3fac := [(1,4)(2,3)(5,6), (1,2)(3,6)(4,5)];
  to3fac := GroupHomomorphismByImages(Group(to3fac), SymmetricGroup(3), to3fac, [(1,2), (2,3)]);
  
  normalize := function(ms)
    local cs;
    cs := List([2,3],i->First(Concatenation(ms[i]), e->not IsZero(e)));
    return [ms[1]*cs[1]*cs[2],ms[2]/cs[1],ms[3]/cs[2]];
  end;
  mss := List(mss, normalize);

  tripForPerm111 := function(sigma)
    local gs, fac_perm, perm;
    gs := List([3,5,1], function (f) 
      local m;
      m := ProjectiveNormalizationMap(TransposedMat(
        Permuted(uss [f^(sigma^local_fac_perm_map)], Inverse(sigma)^term_perm_map111)));
      Assert(0, m <> fail);
      m := Inverse(m) * ProjectiveNormalizationMap(TransposedMat(uss[f]));
      m := m / First(Concatenation(m), e->not IsZero(e));
      return m;
    end);
    fac_perm := sigma^(local_fac_perm_map*to3fac);
    perm := Permutation((), mss, function (ms, e)
      return normalize(TripAction(gs,fac_perm,ms));
    end);
    if perm = fail then
      return fail;
    fi;
    Assert(1, Permutation(perm, Filtered([1..Length(mss)],i-> ForAll(mss[i],m->RankMat(m) = 1))) = sigma ^ term_perm_map111);
    
    # gs are currently normalized, so further transformations will retain the
    # property of being a distinguished representatives mod C^*.
    # We first try to get det close to 1 in absolute value for everything, then 
    # further modify the associated projectively finite order elements into
    # linearly finite order, if possible. This will result in better choices
    # mod C^* in general, in particular ones which might be able to be upgraded to a
    # linear representation later
    det1Mult := function(m)
      local d;
      d := DeterminantMat(m);
      d := d * ComplexConjugate(d);
      return RootInt(DenominatorRat(d),Length(m)*2) / RootInt(NumeratorRat(d),Length(m)*2);
    end;
    gs := List(gs, g -> det1Mult(g)*g);
    if fac_perm = (1,2,3) then
      gs[1] := det1Mult(gs[3] * gs[2] * gs[1]) * gs[1];
    fi;
    if fac_perm = (1,3,2) then
      gs[1] := det1Mult(gs[2] * gs[3] * gs[1]) * gs[1];
    fi;
    if fac_perm = (1,2) then
      gs[2] := det1Mult(TransposedMat(Inverse(gs[3])) * gs[2]) * gs[2];
    fi;
    if fac_perm = (1,3) then
      gs[1] := det1Mult(TransposedMat(Inverse(gs[3])) * gs[1]) * gs[1];
    fi;
    if fac_perm = (2,3) then
      gs[3] := det1Mult(TransposedMat(Inverse(gs[1])) * gs[3]) * gs[3];
    fi;
    return rec(gs:=gs, perm:=perm);
  end;
  
  g := SubgroupProperty(g, e -> tripForPerm111(e) <> fail);
  g := Group(SmallGeneratingSet(g));
  gens := List(GeneratorsOfGroup(g), tripForPerm111);
  if Size(g) > 1 then
    term_perm_map := GroupHomomorphismByImages(g, Group(List(gens, p->p.perm)));
  else
    term_perm_map := GroupHomomorphismByImages(g, g);
  fi;
  
  mono := SmallerDegreePermutationRepresentation(g);
  g := Image(mono);
  fac_perm_map := RestrictedMapping(InverseGeneralMapping(mono),g) * local_fac_perm_map * to3fac;
  term_perm_map := RestrictedMapping(InverseGeneralMapping(mono),g) * term_perm_map;
  
  tripf := function(sigma)
    local p;
    p := tripForPerm111(PreImageElm(mono,sigma));
    Assert(1, sigma^term_perm_map = p.perm);
    return p.gs;
  end;
  return rec(g := g, tripf := tripf, term_perm_map := term_perm_map, 
    fac_perm_map := fac_perm_map); # this is a projective representation with distinguished lifts
end;

# rep can either be a linear representation or a projective representation with distinguished lifts
OrbitStructure := function(rep,mss)
  local act, orbits, res, orbit, term, h, vals;
  act := function( i, e ) return i ^ (e ^ rep.term_perm_map); end;
  orbits := OrbitsDomain(rep.g, [1..Length(mss)], act);
  res := [];
  for orbit in orbits do
    term := orbit[1];
    h := Stabilizer(rep.g, term, act);
    h := Group(SmallGeneratingSet(h));
    Add(res,[h, List(GeneratorsOfGroup(h), function(e) 
      local ms,ms_transform,cur,m;
      ms := mss[term];
      ms_transform := TripAction(rep.tripf(e),e ^ rep.fac_perm_map,ms);
      cur := List([1..3], f-> 
        First(Concatenation(ms_transform[f]),e->not IsZero(e)) / First(Concatenation(ms[f]),e->not IsZero(e)));
      Assert(1,ForAll([1..3], f-> ms[f]*cur[f] = ms_transform[f]));
      m := PermutationMat(e ^ rep.fac_perm_map,3);
      return List(m, r->List([1..3], i-> r[i]*cur[i]));
    end)]);
  od;
  return res;
end;

# The next two functions are inverses of each other, to convert from
# (triple,fac_perm) pairs into a wreath product matrix representation so that
# group operations can be performed and the technology of group homomorphisms
# being defined on generators can be used.
TripToMat := function(trip, fac_perm)
  local to3fac,blocks,bstart,i,M,bi,bj;
  to3fac := [(1,4)(2,3)(5,6), (1,2)(3,6)(4,5)];
  to3fac := GroupHomomorphismByImages(Group(to3fac), SymmetricGroup(3), to3fac, [(1,2), (2,3)]);
  blocks := [TransposedMat(trip[3]), Inverse(trip[1]), TransposedMat(trip[1]), 
      Inverse(trip[2]), TransposedMat(trip[2]), Inverse(trip[3])];
  bstart := Concatenation([0],List(blocks,Length));
  for i in [2..7] do
    bstart[i] := bstart[i] + bstart[i-1];
  od;
  M := ZeroMatrix(bstart[7], bstart[7], blocks[1]);
  for bi in [1..6] do
    for i in [bstart[bi]+1..bstart[bi+1]] do
      bj := bi^PreImageElm(to3fac, fac_perm);
      M[i]{[bstart[bj]+1..bstart[bj+1]]} := blocks[bi][i-bstart[bi]];
    od;
  od;
  return M;
end;

MatToTrip := function(M,uvw)
  local bstart, i, to3fac, perm, gs, fac_perm;
  bstart := [0,uvw[3],uvw[1],uvw[1],uvw[2],uvw[2],uvw[3]];
  for i in [2..7] do
    bstart[i] := bstart[i] + bstart[i-1];
  od;
  to3fac := [(1,4)(2,3)(5,6), (1,2)(3,6)(4,5)];
  to3fac := GroupHomomorphismByImages(Group(to3fac), SymmetricGroup(3), to3fac, [(1,2), (2,3)]);
  perm := PermList(List([1..6], function(i)
    local j;
    j := PositionProperty(M[bstart[i]+1], e->not IsZero(e));
    return First([1..6], i->j <= bstart[i+1]);
  end));
  gs := List([3,5,1],bi ->
    List([bstart[bi]+1..bstart[bi+1]],i -> M[i]{[bstart[bi^perm]+1..bstart[bi^perm+1]]}));
  gs := List(gs, TransposedMat);
  fac_perm := perm ^ to3fac;
  return [gs,fac_perm];
end;

LinearizeRepresentation := function(proj_rep)
  local uvw, i, G, epiToPGL, K, tryRemove, h, Gred, mono;
  uvw := List(proj_rep.tripf(Identity(proj_rep.g)),Length);
  Assert(1,ForAll(Cartesian(proj_rep.g,proj_rep.g), function(p)
    local f;
    f := e -> TripToMat(proj_rep.tripf(e),e^proj_rep.fac_perm_map);
    return IsDiagonalMat(f(p[1])*f(p[2]) *Inverse(f(p[1]*p[2])));
  end));
  Assert(1,ForAll(proj_rep.g, function(e)
    local trip, fac_perm;
    trip := proj_rep.tripf(e);
    fac_perm := e ^ proj_rep.fac_perm_map;
    return MatToTrip(TripToMat(trip, fac_perm), uvw) = [trip, fac_perm];
  end));
  G := Group(List(GeneratorsOfGroup(proj_rep.g), e -> TripToMat(proj_rep.tripf(e), e ^ proj_rep.fac_perm_map)));
  if Size(G) = infinity then
    # This means that when we chose scalings to put the determinant close to
    # the unit circle failed to put the requisite determinants exactly on the
    # unit circle, so the required rescaling is actually irrational. For this
    # to fail here should mean we have found a matrix over the cyclotomics
    # which is projectively of finite order, but which possesses no multiple of
    # finite order defined over the cyclotomics. 
    return fail;
  fi;
  epiToPGL := GroupHomomorphismByImages(G, proj_rep.g);
  
  K := Kernel(epiToPGL);
  tryRemove := List(ConjugacyClassesSubgroups(K),Representative);
  SortBy(tryRemove, h -> Index(K, h));
  for h in tryRemove do
    if IsNormal(G,h) then
      Gred := First(ComplementClassesRepresentatives(G, h));
      if Gred <> fail then
        epiToPGL := RestrictedMapping(epiToPGL, Gred);
        break;
      fi;
    fi;
  od;

  mono := NiceMonomorphism(Source(epiToPGL));
  if IsMatrixGroup(Range(mono)) then
    mono := mono * NiceMonomorphism(Image(mono));
  fi;
  mono := mono * SmallerDegreePermutationRepresentation(Image(mono));
  G := Group(SmallGeneratingSet(Image(mono)));
  epiToPGL := RestrictedMapping(InverseGeneralMapping(mono),G) * epiToPGL;
  return rec(g := G, tripf := e -> MatToTrip(PreImageElm(mono, e),uvw)[1],
    term_perm_map := epiToPGL * proj_rep.term_perm_map,
    fac_perm_map := epiToPGL * proj_rep.fac_perm_map,
    epiToPGL := epiToPGL);
end;

RepresentationMatHom := function(rep)
  local imgs;
  imgs := List(GeneratorsOfGroup(rep.g), e->TripToMat(rep.tripf(e), e ^ rep.fac_perm_map));
  return GroupGeneralMappingByImages(rep.g, GeneratorsOfGroup(rep.g), imgs);
end;

# prep := SymmetryGroupUsingPoints(uss, g, mss);
# rep := LinearizeRepresentation(prep);
# orbits := OrbitStructure(rep,mss);
