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

func TestSubmarineServe(t *testing.T) {
	g := NewGomegaWithT(t)
	r := createSubmarineReconciler()
	submarine, err := MakeSubmarineFromYamlByNamespace("../config/samples/_v1_submarine.yaml", "submarine")
	g.Expect(err).To(BeNil())

	ArtifactBasePath = "../"
	secret1 := r.newSubmarineSeldonSecret(context.TODO(), submarine)
	g.Expect(secret1).NotTo(BeNil())

	// change secret
	submarine.Spec.Minio.AccessKey = "submarine_minio1"
	submarine.Spec.Minio.SecretKey = "submarine_minio2"
	secret2 := r.newSubmarineSeldonSecret(context.TODO(), submarine)
	g.Expect(secret2.StringData["RCLONE_CONFIG_S3_ACCESS_KEY_ID"]).To(Equal("submarine_minio1"))
	g.Expect(secret2.StringData["RCLONE_CONFIG_S3_SECRET_ACCESS_KEY"]).To(Equal("submarine_minio2"))

	// compare
	g.Expect(CompareSecret(secret1, secret2)).To(Equal(false))
}
