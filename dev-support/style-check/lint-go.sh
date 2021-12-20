cd "$(dirname $0)"
cd ../../submarine-cloud-v2

if [ -n "$(go fmt ./...)" ]; then
    echo "Go code is not formatted, please run 'go fmt ./...'." >&2
    exit 1
else
    echo "Go code is formatted"
fi
