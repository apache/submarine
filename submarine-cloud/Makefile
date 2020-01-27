# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get
BINARY_NAME=submarine-operator
BINARY_UNIX=$(BINARY_NAME)_unix
VERSION="0.3.0-SNAPSHOT"
BuildGitBranch=$(git rev-parse --abbrev-ref HEAD)
BuildGitRev=$(git rev-list --count HEAD)
BuildGitCommit=$(git rev-parse HEAD)
LDFLAGS		:= -s -w \
			   -X "github.com/apache/submarine/submarine-cloud/pkg/utils.VERSION=${VERSION}" \
			   -X "github.com/apache/submarine/submarine-cloud/pkg/utils.BuildGitBranch=${BuildGitBranch}" \
			   -X "github.com/apache/submarine/submarine-cloud/pkg/utils.BuildGitRev=${BuildGitRev}" \
			   -X "github.com/apache/submarine/submarine-cloud/pkg/utils.BuildGitCommit=${BuildGitCommit}" \
			   -X "github.com/apache/submarine/submarine-cloud/pkg/utils.BuildTime=$(/bin/date "+%F %T")"

.PHONY: build
build:
	$(GOBUILD) -o ./bin/$(BINARY_NAME) -v cmd/operator/main.go

test:
	$(GOTEST) -v ./...

all: test build

clean:
	$(GOCLEAN)
	rm -f ./$(BINARY_NAME)-$(VERSION)
	rm -f ./$(BINARY_UNIX)-$(VERSION)

fmt:
	@go fmt $(CURDIR)/...

release:
	CGO_ENABLED=0 GOOS=${GOOS} GOARCH=amd64 $(GOBUILD) -a -installsuffix cgo -ldflags "$(LDFLAGS)" -o ./$(BINARY_UNIX)-linux-amd64-$(VERSION) -v cmd/operator/main.go
