let
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/tarball/9ca440cd0acccda37e3e00120918e1165028ff36") { config = {}; overlays = []; };
in
  pkgs.mkShell {
    buildInputs = with pkgs; [
      python3
      sage
      gap-full
    ];
}
		
