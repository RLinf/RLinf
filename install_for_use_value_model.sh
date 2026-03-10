uv pip install scikit-learn
uv pip install lerobot==0.3.3
apt-get update && apt-get install -y ffmpeg liblapack3 libopenblas-base  
# Add value_model to PYTHONPATH in bashrc
VALUE_MODEL_PATH="${VALUE_MODEL_PATH:-/path/to/value_model}"
BASHRC="$HOME/.bashrc"
EXPORT_LINE="export PYTHONPATH=$VALUE_MODEL_PATH:\$PYTHONPATH"

if ! grep -q "$VALUE_MODEL_PATH" "$BASHRC" 2>/dev/null; then
    echo "" >> "$BASHRC"
    echo "# value_model path for RLinf" >> "$BASHRC"
    echo "$EXPORT_LINE" >> "$BASHRC"
    echo "Added value_model to PYTHONPATH in $BASHRC"
else
    echo "value_model already in PYTHONPATH in $BASHRC"
fi

# Also export for current session
export PYTHONPATH=$VALUE_MODEL_PATH:$PYTHONPATH
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