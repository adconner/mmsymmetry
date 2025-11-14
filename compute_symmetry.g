
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
  local fac_perm_map, normalize, tripForPerm111, gens, toTripDomain, tripForPerm;

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
    if SSortedList(mss) = SSortedList(target_mss) then
      perm := SortingPerm(mss) * Inverse(SortingPerm(target_mss));
      perm := Inverse(SortingPerm(List(Cartesian([1..Length(mss)],[1..3]),p -> [p[1]^perm, p[2]^fac_perm])));
      return [gs, fac_perm, perm];
    fi;
    return fail;
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
  return [g, tripForPerm];
end;

OrbitStructure := function(g,tripForPerm,mss)
  local maps, fac_perm_map, orbits, res, orbit, term, h, vals;
  orbits := OrbitsDomain(g, List([1..Length(mss)], i-> [3*(i-1)+1..3*(i-1)+3]), OnSets);
  res := [];
  for orbit in orbits do
    term := orbit[1];
    h := Stabilizer(g, term, OnSets);
    vals := List(ConjugacyClasses(h), function(cl) 
      local p,ms,ms_transform,res;
      p := tripForPerm(Representative(cl));
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
