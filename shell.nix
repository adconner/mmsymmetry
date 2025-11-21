let
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/tarball/9ca440cd0acccda37e3e00120918e1165028ff36") { 
    config = {
      allowUnfree = true;
    }; 
    overlays = [
    (self: super: {
      gap = super.gap.override { packageSet = "full"; };
      sage = super.sage.override (prev :{ 
        requireSageTests = false; 
        extraPythonPackages = (python-pkgs: with python-pkgs; [
          jax
          jax-cuda12-plugin
          optax
        ]);
      });
    })
  ]; };
in
  pkgs.mkShell {
    buildInputs = with pkgs; [
      sage
    ];
    shellHook = ''
      cat > .enable_bliss.g <<EOL
      GRAPE_NAUTY := false;
      GRAPE_BLISS_EXE := "${pkgs.bliss}/bin/bliss";
      EOL
    '';
}
		
