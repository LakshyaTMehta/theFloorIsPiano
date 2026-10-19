{
    description = "voise dev shell";
    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    };
    outputs = inputs@{ self, nixpkgs }:
    let
        system = "x86_64-linux";
        pkgs = import nixpkgs {
            inherit system;
        };
    in
    {
        devShells."${system}".default = pkgs.mkShell {
            packages = with pkgs; [
                uv
                python3
                stdenv.cc.cc.lib
                zlib
                qt5.qtbase
                wl-clipboard
                pulseaudio
            ];

            env = {

                QT_QPA_PLATFORM_PLUGIN_PATH =
                    "${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins/platforms";
                
                QT_QPA_PLATFORM = "xcb";

                LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs;[
                    stdenv.cc.cc.lib
                    zlib
                    libxcb
                    libSM
                    libICE
                    libGL
                    glib
                    pulseaudio
                ]);
            };

            shellHook = ''
            '';
        };
    };
}

