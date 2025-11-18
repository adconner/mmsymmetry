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
  local local_fac_perm_map, term_perm_map111, to3fac, normalize, num111,
        tripForPerm111, gens, term_perm_map, tripf, det1Mult;

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

  num111 := Length(uss[1]);
  tripForPerm111 := function(sigma)
    local gs, fac_perm, target_mss, perm;
    gs := List([1..3], function (f) 
      local m, lfac;
      lfac := 2*(f mod 3) + 1;
      m := ProjectiveNormalizationMap(TransposedMat(
        Permuted(uss [lfac^(Inverse(sigma)^local_fac_perm_map)], sigma^term_perm_map111)));
      if m <> fail then
        m := Inverse(m) * ProjectiveNormalizationMap(TransposedMat(uss[lfac]));
        m := m / First(Concatenation(m), e->not IsZero(e));
        return m;
      fi;
      return fail;
    end);
    if ForAny(gs, g-> g = fail) then
      return fail;
    fi;
    fac_perm := sigma^(local_fac_perm_map*to3fac);
    target_mss := List(mss, ms -> normalize(TripAction(gs,fac_perm,ms)));
    if SSortedList(mss) <> SSortedList(target_mss) then
      return fail;
    fi;
    perm := SortingPerm(target_mss) * Inverse(SortingPerm(mss));
    
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
  gens := List(GeneratorsOfGroup(g), tripForPerm111);

  term_perm_map := GroupHomomorphismByImages(g, Group(List(gens, p->p.perm)));
  tripf := function(sigma)
    local p;
    p := tripForPerm111(sigma);
    Assert(0, sigma^term_perm_map = p.perm);
    return p.gs;
  end;
  return rec(g := g, tripf := tripf, term_perm_map := term_perm_map, 
    fac_perm_map := local_fac_perm_map * to3fac); # this is a projective representation with distinguished lifts
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
    vals := List(ConjugacyClasses(h), function(cl) 
      local ms,ms_transform,cur;
      ms := mss[term];
      ms_transform := TripAction(rep.tripf(Representative(cl)),Representative(cl) ^ rep.fac_perm_map,ms);
      cur := List([1..3], f-> 
        First(Concatenation(ms_transform[f]),e->not IsZero(e)) / First(Concatenation(ms[f]),e->not IsZero(e)));
      Assert(0,ForAll([1..3], f-> ms[f]*cur[f] = ms_transform[f]));
      return cur;
    end);
    Add(res,[h, List([1..3], f-> ClassFunction(h,List(vals, p -> p[f])))]);
  od;
  return res;
end;

LinearizeRepresentation := function(proj_rep)
  local to3fac, tripToMat, matToTrip, G, epiToPGL, K, tryRemove, h, Gred, mono;
  to3fac := [(1,4)(2,3)(5,6), (1,2)(3,6)(4,5)];
  to3fac := GroupHomomorphismByImages(Group(to3fac), SymmetricGroup(3), to3fac,  [(1,2), (2,3)]);
  tripToMat := function(trip, fac_perm)
    local dblocks;
    dblocks := [TransposedMat(trip[3]), Inverse(trip[1]), TransposedMat(trip[1]), 
        Inverse(trip[2]), TransposedMat(trip[2]), Inverse(trip[3])];
    return BlockMatrix(List([1..6], i-> [i, i^PreImageElm(to3fac,fac_perm), dblocks[i]]),6,6);
  end;
  matToTrip := function(M)
    local n, perm, gs, fac_perm;
    n := Length(M) / 6;
    perm := PermList(List([1..6], i-> QuoInt(PositionProperty(M[(i-1)*n+1], e->not IsZero(e))-1,n)+1));
    gs := List([1..3],f -> TransposedMat(List([1..n], i->
        M[ (f mod 3)*2*n + i ]{[((2*(f mod 3)+1)^perm-1)*n+1..((2*(f mod 3)+1)^perm)*n]})));
    return gs;
    #fac_perm := perm ^ to3fac;
    #return [gs,fac_perm];
  end;
  G := Group(List(GeneratorsOfGroup(proj_rep.g), e -> tripToMat(proj_rep.tripf(e), e ^ proj_rep.fac_perm_map)));
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
  G := Image(mono);
  epiToPGL := RestrictedMapping(InverseGeneralMapping(mono),G) * epiToPGL;
  return rec(g := Image(mono), tripf := e -> matToTrip(PreImageElm(mono, e)), 
    term_perm_map := epiToPGL * proj_rep.term_perm_map,
    fac_perm_map := epiToPGL * proj_rep.fac_perm_map,
    epiToPGL := epiToPGL);
end;

