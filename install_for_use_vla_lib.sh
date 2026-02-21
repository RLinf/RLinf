uv pip install scikit-learn
uv pip install lerobot==0.3.3
apt-get update && apt-get install -y ffmpeg liblapack3 libopenblas-base  
# Add vla_lib to PYTHONPATH in bashrc
VLA_LIB_PATH="${VLA_LIB_PATH:-/path/to/vla_lib}"
BASHRC="$HOME/.bashrc"
EXPORT_LINE="export PYTHONPATH=$VLA_LIB_PATH:\$PYTHONPATH"

if ! grep -q "$VLA_LIB_PATH" "$BASHRC" 2>/dev/null; then
    echo "" >> "$BASHRC"
    echo "# vla_lib path for RLinf" >> "$BASHRC"
    echo "$EXPORT_LINE" >> "$BASHRC"
    echo "Added vla_lib to PYTHONPATH in $BASHRC"
else
    echo "vla_lib already in PYTHONPATH in $BASHRC"
fi

# Also export for current session
export PYTHONPATH=$VLA_LIB_PATH:$PYTHONPATH
echo "PYTHONPATH updated for current session"

# Create compatibility shim for lerobot 0.3.x
# openpi uses lerobot.common.datasets (old API), but lerobot 0.3.x uses lerobot.datasets (new API)
LEROBOT_PATH=$(python -c "import lerobot; import os; print(os.path.dirname(lerobot.__file__))")
echo "Creating lerobot.common compatibility shim at: $LEROBOT_PATH/common"

mkdir -p "$LEROBOT_PATH/common/datasets"

cat > "$LEROBOT_PATH/common/__init__.py" << 'EOF'
# Compatibility shim for lerobot 0.3.x
# Redirects lerobot.common.* to lerobot.*
EOF

cat > "$LEROBOT_PATH/common/datasets/__init__.py" << 'EOF'
# Compatibility shim for lerobot 0.3.x
# Redirects lerobot.common.datasets.* to lerobot.datasets.*
from lerobot.datasets import *
EOF

cat > "$LEROBOT_PATH/common/datasets/lerobot_dataset.py" << 'EOF'
# Compatibility shim for lerobot 0.3.x
# Redirects lerobot.common.datasets.lerobot_dataset to lerobot.datasets.lerobot_dataset
from lerobot.datasets.lerobot_dataset import *
EOF

echo "lerobot.common compatibility shim created successfully"