TARGET=$1
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

if [ -z "$TARGET" ]; then
  TARGET="en"
fi

sphinx-autobuild source-$TARGET build/html
