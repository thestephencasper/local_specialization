echo "Installing Python dependencies..."
# PIP_USER=yes  # install as --user
pipenv install --system

echo "Patching scipy and sklearn..."
python -m site | grep /usr/local/lib/python3.7/dist-packages || { echo "Cannot find dist-packages directory for patching of scipy and sklearn"; exit 1; }
cp -r devops/patches/* /usr/local/lib/python3.7/dist-packages
