module submarine-cloud-v2/test/e2e

go 1.16

require (
	github.com/stretchr/testify v1.7.0
	k8s.io/apimachinery v0.21.1
	k8s.io/client-go v0.21.1 // indirect
	submarine-cloud-v2 v0.0.0-00010101000000-000000000000 // indirect
	submarine-cloud-v2/test/e2e/framework v0.0.0-00010101000000-000000000000
)

replace (
	github.com/abbot/go-http-auth => github.com/containous/go-http-auth v0.4.1-0.20200324110947-a37a7636d23e
	github.com/docker/distribution => github.com/docker/distribution v0.0.0-20191216044856-a8371794149d
	github.com/docker/docker => github.com/moby/moby v17.12.0-ce-rc1.0.20200618181300-9dc6525e6118+incompatible
	github.com/go-check/check => github.com/containous/check v0.0.0-20170915194414-ca0bf163426a
	github.com/gorilla/mux => github.com/containous/mux v0.0.0-20181024131434-c33f32e26898
	github.com/mailgun/minheap => github.com/containous/minheap v0.0.0-20190809180810-6e71eb837595
	github.com/mailgun/multibuf => github.com/containous/multibuf v0.0.0-20190809014333-8b6c9a7e6bba
	submarine-cloud-v2 => ../../
	submarine-cloud-v2/test/e2e/framework => ./framework
)
