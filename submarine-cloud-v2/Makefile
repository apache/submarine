.PHONY: image
image:
	GOOS=linux go build -o submarine-operator
	docker build -t submarine-operator .
	go build -o submarine-operator
	touch $@

.PHONY: test-unit
test-unit:
	go test
