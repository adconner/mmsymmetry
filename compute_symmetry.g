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

LoadPackage("gauss");
SymmetryGroupUsingPoints := function(us, g, frame_ixs, mss)
  local fac_perm_map, normalize, tripForPerm, gens;

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

  tripForPerm := function(sigma)
    local gs, fac_perm, target_mss;
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
    target_mss := List(mss, function (ms)
      ms := [gs[3]*ms[1]*Inverse(gs[1]), gs[1]*ms[2]*Inverse(gs[2]), gs[2]*ms[3]*Inverse(gs[3])];
      ms := Permuted(ms, fac_perm);
      if SignPerm(fac_perm) = -1 then
        ms := List(ms,TransposedMat);
      fi;
      return normalize(ms);
    end);
    if SSortedList(mss) = SSortedList(target_mss) then
      return [gs, fac_perm, Inverse(Sortex(target_mss)) * Sortex(mss)];
    fi;
    return fail;
  end;
  
  # g := Kernel(fac_perm_map);
  g := SubgroupProperty(g, e -> tripForPerm(e) <> fail);
  gens := List(GeneratorsOfGroup(g), tripForPerm);

  g := DirectProduct(SymmetricGroup(Length(mss)), SymmetricGroup(3));
  g := Group(List(gens, p-> p[3]^Embedding(g,1) * p[2]^Embedding(g,2)));
  gens := List(gens, p -> [p[1],p[2]]);
  return [g, gens];
end;

