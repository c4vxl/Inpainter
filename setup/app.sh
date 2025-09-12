#!/bin/bash
sudo -v

DIR="$HOME/.local/share/Inpainter/"

echo ">> Generating data dir"
mkdir -p "$DIR" &> /dev/null

echo ">> Downloading application"
git clone https://github.com/c4vxl/Inpainter "$DIR" &> /dev/null

echo ">> Entering app dir"
cd "$DIR"

echo ">> Installing uv..."
python -m pip install uv &> /dev/null

echo ">> Creating virtual environment"
python -m venv ".venv"

echo "  | Activating env"
source .venv/bin/activate

echo "  | Installing packages"
uv pip install -r requirements.txt

echo "  | Creating launcher"
cat <<'EOF' | sudo tee /usr/bin/inpainter > /dev/null
#!/bin/bash
cd "$HOME/.local/share/Inpainter/"
source .venv/bin/activate
python src/server.py "$@"
EOF
sudo chmod +x /usr/bin/inpainter

cat <<'EOF' | sudo tee /usr/bin/inpainter-app > /dev/null
#!/bin/bash
cd "$HOME/.local/share/Inpainter/"
source .venv/bin/activate
python src/app.py
EOF
sudo chmod +x /usr/bin/inpainter-app

cat <<'EOF' | sudo tee /usr/bin/inpainter-uninstall > /dev/null
#!/bin/bash
sudo rm -R "$HOME/.local/share/Inpainter/"
sudo rm /usr/bin/inpainter-app
sudo rm /usr/bin/inpainter-uninstall
sudo rm $HOME/.local/share/applications/inpainter.desktop
EOF
sudo chmod +x /usr/bin/inpainter-uninstall

echo ">> Creating desktop entry"
echo "[Desktop Entry]"                                                       > $HOME/.local/share/applications/inpainter.desktop
echo "Name=Inpainter"                                                        >> $HOME/.local/share/applications/inpainter.desktop
echo "Comment=A tool that implements a selective image inpainting pipeline." >> $HOME/.local/share/applications/inpainter.desktop
echo "Icon=$DIR/resources/logo.png"                                          >> $HOME/.local/share/applications/inpainter.desktop
echo "Exec=/usr/bin/inpainter-app"                                           >> $HOME/.local/share/applications/inpainter.desktop
echo "Terminal=False"                                                        >> $HOME/.local/share/applications/inpainter.desktop
echo "Type=Application"                                                      >> $HOME/.local/share/applications/inpainter.desktop
echo "StartupWMClass=Inpainter"                                              >> $HOME/.local/share/applications/inpainter.desktop

echo ">>> All done."