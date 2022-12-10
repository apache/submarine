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
	submarineapacheorgv1alpha1 "github.com/apache/submarine/submarine-cloud-v3/api/v1alpha1"
	"testing"

	. "github.com/apache/submarine/submarine-cloud-v3/controllers/util"
	. "github.com/onsi/gomega"
)

func TestSubmarineMlflow(t *testing.T) {
	g := NewGomegaWithT(t)
	r := createSubmarineReconciler()
	submarine, err := MakeSubmarineFromYamlByNamespace("../config/samples/_v1alpha1_submarine.yaml", "submarine")
	g.Expect(err).To(BeNil())

	ArtifactBasePath = "../"
	deployment1 := r.newSubmarineMlflowDeployment(context.TODO(), submarine)
	g.Expect(deployment1).NotTo(BeNil())
	g.Expect(deployment1.Spec.Template.Spec.Containers[0].Image).To(Equal("apache/submarine:mlflow-" + submarine.Spec.Version))

	// test change params
	submarine.Spec.Mlflow.Image = "harbor.com/apache/submarine/mlflow-" + submarine.Spec.Version
	submarine.Spec.Common = &submarineapacheorgv1alpha1.SubmarineCommon{
		Image: submarineapacheorgv1alpha1.CommonImage{
			McImage:      "harbor.com/minio/mc",
			BusyboxImage: "harbor.com/busybox:1.28",
			PullSecrets:  []string{"pull-secret"},
		},
	}
	deployment2 := r.newSubmarineMlflowDeployment(context.TODO(), submarine)
	g.Expect(deployment2.Spec.Template.Spec.Containers[0].Image).To(Equal("harbor.com/apache/submarine/mlflow-" + submarine.Spec.Version))
	g.Expect(deployment2.Spec.Template.Spec.InitContainers[0].Image).To(Equal("harbor.com/busybox:1.28"))
	g.Expect(deployment2.Spec.Template.Spec.InitContainers[1].Image).To(Equal("harbor.com/minio/mc"))
	g.Expect(deployment2.Spec.Template.Spec.ImagePullSecrets[0].Name).To(Equal("pull-secret"))

	// test compare
	g.Expect(r.compareMlflowDeployment(deployment1, deployment2)).To(Equal(false))
}

func TestSubmarineMlflowOpenshift(t *testing.T) {
	g := NewGomegaWithT(t)
	r := createSubmarineReconciler(&SubmarineReconciler{SeldonIstioEnable: true, ClusterType: "openshift"})
	submarine, _ := MakeSubmarineFromYamlByNamespace("../config/samples/_v1alpha1_submarine.yaml", "submarine")

	ArtifactBasePath = "../"
	deployment := r.newSubmarineMlflowDeployment(context.TODO(), submarine)
	g.Expect(deployment).NotTo(BeNil())
	g.Expect(*deployment.Spec.Template.Spec.InitContainers[0].SecurityContext.RunAsUser).To(Equal(int64(istioSidecarUid)))
	g.Expect(*deployment.Spec.Template.Spec.InitContainers[1].SecurityContext.RunAsUser).To(Equal(int64(istioSidecarUid)))
}
