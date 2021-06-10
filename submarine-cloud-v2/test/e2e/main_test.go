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

	operatorFramework "submarine-cloud-v2/test/e2e/framework"
)


var (
	framework *operatorFramework.Framework
)

func TestMain(m *testing.M) {
	kubeconfig := flag.String("kubeconfig", os.Getenv("HOME")+"/.kube/config", "Path to a kubeconfig. Only required if out-of-cluster.")
	opImage := flag.String("operator-image", "", "operator image, e.g. image:tag")
	opImagePullPolicy := flag.String("operator-image-pullPolicy", "Never", "pull policy, e.g. Always")
	ns := flag.String("namespace", "default", "e2e test operator namespace") 
	submarineTestNamespace := flag.String("submarine-test-namespace", "submarine-admin", "e2e test submarine namespace")
	flag.Parse()
	
	var (
		err error
		exitCode int
	)

	if framework, err = operatorFramework.New(*ns, *submarineTestNamespace, *kubeconfig, *opImage, *opImagePullPolicy); err != nil {
		log.Fatalf("Error setting up framework: %v", err)
	}

	operatorFramework.SubmarineTestNamespace = *submarineTestNamespace

	exitCode = m.Run()

	// if err := framework.Teardown(); err != nil {
	// 	log.Fatalf("Failed to tear down framework :%v\n", err)
	// }
	os.Exit(exitCode)
}


