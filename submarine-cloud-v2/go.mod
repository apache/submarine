module github.com/apache/submarine/submarine-cloud-v2

go 1.16

require (
	github.com/fatih/color v1.7.0
	github.com/gofrs/flock v0.8.0
	github.com/pkg/errors v0.9.1
	github.com/stretchr/testify v1.7.0
	github.com/traefik/traefik/v2 v2.4.8
	golang.org/x/sync v0.0.0-20201207232520-09787c993a3a
	gopkg.in/yaml.v2 v2.4.0
	helm.sh/helm/v3 v3.5.3
	k8s.io/api v0.20.4
	k8s.io/apimachinery v0.20.4
	k8s.io/client-go v0.20.4
	k8s.io/code-generator v0.21.2
	k8s.io/klog/v2 v2.8.0
)

replace (
	github.com/abbot/go-http-auth => github.com/containous/go-http-auth v0.4.1-0.20200324110947-a37a7636d23e
	github.com/docker/distribution => github.com/docker/distribution v0.0.0-20191216044856-a8371794149d
	github.com/docker/docker => github.com/moby/moby v17.12.0-ce-rc1.0.20200618181300-9dc6525e6118+incompatible
	github.com/go-check/check => github.com/containous/check v0.0.0-20170915194414-ca0bf163426a
	github.com/gorilla/mux => github.com/containous/mux v0.0.0-20181024131434-c33f32e26898
	github.com/mailgun/minheap => github.com/containous/minheap v0.0.0-20190809180810-6e71eb837595
	github.com/mailgun/multibuf => github.com/containous/multibuf v0.0.0-20190809014333-8b6c9a7e6bba
)
