/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package utils

import (
	"fmt"
	"time"
)

// BUILDTIME should be populated by at build time: -ldflags "-w -X github.com/apache/submarine/submarine-cloud/pkg/utils.BUILDTIME=${DATE}
// with for example DATE=$(shell date +%Y-%m-%d/%H:%M:%S )   (pay attention not to use space!)
var BUILDTIME string

// TAG should be populated by at build time: -ldflags "-w -X github.com/apache/submarine/submarine-cloud/pkg/utils.TAG=${TAG}
// with for example TAG=$(shell git tag|tail -1)
var TAG string

// COMMIT should be populated by at build time: -ldflags "-w -X github.com/apache/submarine/submarine-cloud/pkg/utils.COMMIT=${COMMIT}
// with for example COMMIT=$(shell git rev-parse HEAD)
var COMMIT string

// VERSION should be populated by at build time: -ldflags "-w -X github.com/apache/submarine/submarine-cloud/pkg/utils.VERSION=${VERSION}
// with for example VERSION=$(shell git rev-parse --abbrev-ref HEAD)
var VERSION string

// BuildInfos returns builds information
func BuildInfos() {
	fmt.Println("Program started at: " + time.Now().String())
	fmt.Println("BUILDTIME=" + BUILDTIME)
	fmt.Println("TAG=" + TAG)
	fmt.Println("COMMIT=" + COMMIT)
	fmt.Println("VERSION=" + VERSION)
}
