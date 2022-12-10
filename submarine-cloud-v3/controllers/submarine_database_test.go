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

	. "github.com/apache/submarine/submarine-cloud-v3/controllers/util"
	. "github.com/onsi/gomega"
)

func TestSubmarineDatabase(t *testing.T) {
	g := NewGomegaWithT(t)
	r := createSubmarineReconciler(&SubmarineReconciler{Namespace: "submarine"})
	submarine, err := MakeSubmarineFromYamlByNamespace("../config/samples/_v1alpha1_submarine.yaml", "submarine")
	g.Expect(err).To(BeNil())

	ArtifactBasePath = "../"
	statefulset1 := r.newSubmarineDatabaseStatefulSet(context.TODO(), submarine)
	g.Expect(statefulset1).NotTo(BeNil())

	// change secret
	submarine.Spec.Database.MysqlRootPasswordSecret = "mysql-password-secret"
	statefulset2 := r.newSubmarineDatabaseStatefulSet(context.TODO(), submarine)
	g.Expect(statefulset2.Spec.Template.Spec.Containers[0].Env[0].ValueFrom.SecretKeyRef.Name).To(Equal("mysql-password-secret"))

	// compare
	g.Expect(r.compareDatabaseStatefulset(statefulset1, statefulset2)).To(Equal(false))
}
