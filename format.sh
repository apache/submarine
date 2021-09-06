# set -euxo pipefail

# FWDIR="$(cd "$(dirname "$0")"; pwd)"
# cd "$FWDIR"
# cd ..
isort website/ submarine-sdk/ dev-support/
black website/ submarine-sdk/ dev-support/
# Autoformat code
# yapf -i submarine/**/*.py tests/**/*.py
# Sort imports
# isort submarine/**/*.py tests/**/*.py

# set +euxo pipefail