FrameMap := function(M)
  local i,res,vec,inv,d;
  res := M{[1..Length(M)-1]};
  vec := SolutionMat(res, M[Length(M)]);
  if vec = fail then return fail; fi;
  for i in [1..Length(vec)] do
    if IsZero(vec[i]) then return fail; fi;
    res[i] := vec[i] * res[i];
  od;
  return res;
end;

TripAction := function (gs,fac_perm,ms)
  ms := [gs[3]*ms[1]*Inverse(gs[1]), gs[1]*ms[2]*Inverse(gs[2]), gs[2]*ms[3]*Inverse(gs[3])];
  ms := Permuted(ms, fac_perm);
  if SignPerm(fac_perm) = -1 then
    ms := List(ms,TransposedMat);
  fi;
  return ms;
end;


LoadPackage("gauss");
SymmetryGroupUsingPoints := function(us, g, frame_ixs, mss)
  local fac_perm_map, normalize, tripForPerm111, gens, toTripDomain, tripForPerm, det1Mult;

  fac_perm_map := ActionHomomorphism(g, 
    [Filtered([1..Length(us)],i->i mod 6 in [1,2]),
     Filtered([1..Length(us)],i->i mod 6 in [3,4]),
     Filtered([1..Length(us)],i->i mod 6 in [5,0])], OnSets);
  
  normalize := function(ms)
    local cs;
    cs := List([2,3],i->First(Concatenation(ms[i]), e->not IsZero(e)));
    return [ms[1]*cs[1]*cs[2],ms[2]/cs[1],ms[3]/cs[2]];
  end;
  mss := List(mss, normalize);

  tripForPerm111 := function(sigma)
    local gs, fac_perm, target_mss, perm;
    gs := List(frame_ixs, function (fr) 
      local m;
      m := FrameMap(us{ List(fr,i->i^sigma) });
      if m <> fail then
        m := TransposedMat(m) * Inverse(TransposedMat(FrameMap(us{fr})));
        m := m / First(Concatenation(m), e->not IsZero(e));
        return m;
      fi;
      return fail;
    end);
    if ForAny(gs, g-> g = fail) then
      return fail;
    fi;
    fac_perm := sigma ^ fac_perm_map;
    target_mss := List(mss, ms -> normalize(TripAction(gs,fac_perm,ms)));
    if SSortedList(mss) <> SSortedList(target_mss) then
      return fail;
    fi;
    perm := SortingPerm(mss) * Inverse(SortingPerm(target_mss));
    perm := Inverse(SortingPerm(List(Cartesian([1..Length(mss)],[1..3]),p -> [p[1]^perm, p[2]^fac_perm])));
    
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
    return [gs, fac_perm, perm];
  end;
  
  g := SubgroupProperty(g, e -> tripForPerm111(e) <> fail);
  gens := List(GeneratorsOfGroup(g), tripForPerm111);

  toTripDomain := GroupHomomorphismByImages(Group(List(gens, p->p[3])), List(gens, p->p[3]), GeneratorsOfGroup(g));
  g := Source(toTripDomain);
  tripForPerm := function(sigma)
    local p;
    p := tripForPerm111(sigma ^ toTripDomain);
    Assert(0, sigma = p[3]);
    return [p[1],p[2]];
  end;
  return rec(g := g, tripf := tripForPerm); # this is a projective representation with distinguished lifts
end;

# rep can either be a linear representation or a projective representation with distinguished lifts
OrbitStructure := function(rep,mss)
  local maps, fac_perm_map, orbits, res, orbit, term, h, vals;
  orbits := OrbitsDomain(rep.g, List([1..Length(mss)], i-> [3*(i-1)+1..3*(i-1)+3]), OnSets);
  res := [];
  for orbit in orbits do
    term := orbit[1];
    h := Stabilizer(rep.g, term, OnSets);
    vals := List(ConjugacyClasses(h), function(cl) 
      local p,ms,ms_transform,res;
      p := rep.tripf(Representative(cl));
      ms := mss[(term[1]-1)/3+1];
      ms_transform := TripAction(p[1],p[2],ms);
      res := List([1..3], f-> 
        First(Concatenation(ms_transform[f]),e->not IsZero(e)) / First(Concatenation(ms[f]),e->not IsZero(e)));
      Assert(0,ForAll([1..3], f-> ms[f]*res[f] = ms_transform[f]));
      return res;
    end);
    Append(res,[h, List([1..3], f-> ClassFunction(h,List(vals, p -> p[f])))]);
  od;
  return res;
end;

LinearizeRepresentation := function(proj_rep)
  local to6fac, tripToMat, matToTrip, G, epiToPGL, K, tryRemove, h, Gred, mono;
  to6fac := [(1,4)(2,3)(5,6), (1,2)(3,6)(4,5)];
  to6fac := GroupHomomorphismByImages(SymmetricGroup(3), Group(to6fac), [(1,2), (2,3)], to6fac);
  tripToMat := function(trip)
    local dblocks;
    dblocks := [trip[1][3], Inverse(TransposedMat(trip[1][1])), trip[1][1], 
        Inverse(TransposedMat(trip[1][2])), trip[1][2], Inverse(TransposedMat(trip[1][3]))];
    return BlockMatrix(List([1..6], i-> [i^(trip[2]^to6fac), i, dblocks[i]]),6,6);
  end;
  matToTrip := function(M)
    local n, perm, gs, fac_perm;
    n := Length(M) / 6;
    perm := PermList(List([1..6], i-> QuoInt(PositionProperty(M[(i-1)*n+1], e->not IsZero(e))-1,n)+1));
    gs := List([1..3],f -> List([1..n], i->
        M[ n*((2*(f mod 3)+1)^perm-1)+i ]{[2*(f mod 3)*n+1..(2*(f mod 3)+1)*n]}));
    fac_perm := Inverse(PreImageElm(to6fac, perm));
    return [gs,fac_perm];
  end;
  G := Group(List(GeneratorsOfGroup(proj_rep.g), e -> tripToMat(proj_rep.tripf(e))));
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
  return rec(g := Image(mono), tripf := e -> matToTrip(PreImageElm(mono, e)), epiToPGL := epiToPGL);
end;

