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
	submarineapacheorgv1 "github.com/apache/submarine/submarine-cloud-v3/api/v1"
	"testing"

	. "github.com/apache/submarine/submarine-cloud-v3/controllers/util"
	. "github.com/onsi/gomega"
)

func TestSubmarineTensorboard(t *testing.T) {
	g := NewGomegaWithT(t)
	r := createSubmarineReconciler()
	submarine, err := MakeSubmarineFromYamlByNamespace("../config/samples/_v1_submarine.yaml", "submarine")
	g.Expect(err).To(BeNil())

	ArtifactBasePath = "../"
	deployment1 := r.newSubmarineTensorboardDeployment(context.TODO(), submarine)
	g.Expect(deployment1).NotTo(BeNil())

	// test change params
	submarine.Spec.Tensorboard.Image = "harbor.com/tensorflow/tensorflow:1.11.0"
	submarine.Spec.Common = &submarineapacheorgv1.SubmarineCommon{
		Image: submarineapacheorgv1.CommonImage{
			PullSecrets: []string{"pull-secret"},
		},
	}
	deployment2 := r.newSubmarineTensorboardDeployment(context.TODO(), submarine)
	g.Expect(deployment2.Spec.Template.Spec.Containers[0].Image).To(Equal("harbor.com/tensorflow/tensorflow:1.11.0"))
	g.Expect(deployment2.Spec.Template.Spec.ImagePullSecrets[0].Name).To(Equal("pull-secret"))

	// test compare
	g.Expect(r.compareTensorboardDeployment(deployment1, deployment2)).To(Equal(false))
}
