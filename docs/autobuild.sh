TARGET=$1

if [ -z "$TARGET" ]; then
  TARGET="en"
fi

sphinx-autobuild source-$TARGET build/html