#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

all: # The default target

charts:
	cp -r ../helm-charts/submarine/charts  .

.PHONY: all
all: charts
	go build -o submarine-operator

.PHONY: api
api:
	./hack/update-codegen.sh

.PHONY: image
image: charts
	GOOS=linux go build -o submarine-operator
	docker build -t submarine-operator .
	go build -o submarine-operator

.PHONY: test-unit
test-unit:
	go test
