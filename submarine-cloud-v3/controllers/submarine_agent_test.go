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

package controllers

import (
	"context"
	"testing"

	submarineapacheorgv1 "github.com/apache/submarine/submarine-cloud-v3/api/v1"

	. "github.com/apache/submarine/submarine-cloud-v3/controllers/util"
	. "github.com/onsi/gomega"
)

func TestSubmarineAgent(t *testing.T) {
	g := NewGomegaWithT(t)
	r := createSubmarineReconciler(&SubmarineReconciler{Namespace: "submarine"})
	submarine, err := MakeSubmarineFromYamlByNamespace("../config/samples/_v1_submarine.yaml", "submarine")
	g.Expect(err).To(BeNil())

	ArtifactBasePath = "../"
	deployment1 := r.newSubmarineAgentDeployment(context.TODO(), submarine)
	g.Expect(deployment1).NotTo(BeNil())

	// change image
	submarine.Spec.Agent = &submarineapacheorgv1.SubmarineAgent{
		Image: "apache/submarine:agent",
	}
	deployment2 := r.newSubmarineAgentDeployment(context.TODO(), submarine)
	g.Expect(deployment2.Spec.Template.Spec.Containers[0].Image).To(Equal("apache/submarine:agent"))

	// compare
	g.Expect(r.CompareAgentDeployment(deployment1, deployment2)).To(Equal(false))
}
