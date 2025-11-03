TARGET=$1
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

if [ -z "$TARGET" ]; then
  TARGET="en"
fi

sphinx-autobuild source-$TARGET build/html