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

package e2e

import (
	"flag"
	"log"
	"os"
	"testing"
	"time"

	v1alpha1 "github.com/apache/submarine/submarine-cloud-v2/pkg/apis/submarine/v1alpha1"

	operatorFramework "github.com/apache/submarine/submarine-cloud-v2/test/e2e/framework"
	"github.com/stretchr/testify/assert"
)

var (
	framework *operatorFramework.Framework
)

// Wait for test job to finish.
var TIMEOUT = 1200 * time.Second
var INTERVAL = 2 * time.Second

var STATES = [4]string{
	"",
	"CREATING",
	"RUNNING",
	"FAILED",
}

func GetJobStatus(t *testing.T, submarineNs, submarineName string) v1alpha1.SubmarineStateType {
	submarine, err := operatorFramework.GetSubmarine(framework.SubmarineClient, submarineNs, submarineName)
	assert.Equal(t, nil, err)
	return submarine.Status.SubmarineState.State
}

func TestMain(m *testing.M) {
	kubeconfig := flag.String("kubeconfig", os.Getenv("HOME")+"/.kube/config", "Path to a kubeconfig. Only required if out-of-cluster.")
	opImage := flag.String("operator-image", "", "operator image, e.g. image:tag")
	opImagePullPolicy := flag.String("operator-image-pullPolicy", "Never", "pull policy, e.g. Always")
	ns := flag.String("namespace", "default", "e2e test operator namespace")
	flag.Parse()

	var (
		err      error
		exitCode int
	)

	if framework, err = operatorFramework.New(*ns, *kubeconfig, *opImage, *opImagePullPolicy); err != nil {
		log.Fatalf("Error setting up framework: %+v", err)
	}

	exitCode = m.Run()

	if err := framework.Teardown(); err != nil {
		log.Fatalf("Failed to tear down framework :%v\n", err)
	}
	os.Exit(exitCode)
}
