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
	"context"
	"testing"

	"github.com/apache/submarine/submarine-cloud-v2/pkg/apis/submarine/v1alpha1"
	operatorFramework "github.com/apache/submarine/submarine-cloud-v2/test/e2e/framework"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"

	"github.com/stretchr/testify/assert"
)

// create & delete submarine custom resource with yaml
func TestSubmitSubmarineCustomResourceYaml(t *testing.T) {
	t.Log("[TestSubmitSubmarineCustomResourceYaml]")

	// create a test namespace
	submarineNs := "submarine-test-submit-custom-resource"
	_, err := framework.KubeClient.CoreV1().Namespaces().Create(context.TODO(), &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: submarineNs,
		},
	}, metav1.CreateOptions{})
	if apierrors.IsAlreadyExists(err) {
		_, err = framework.KubeClient.CoreV1().Namespaces().Get(context.TODO(), submarineNs, metav1.GetOptions{})
	}
	assert.Equal(t, nil, err)

	submarine, err := operatorFramework.MakeSubmarineFromYaml("../../artifacts/examples/example-submarine.yaml")
	assert.Equal(t, nil, err)
	submarineName := submarine.Name

	// create submarine
	t.Logf("[Create] Submarine %s/%s", submarineNs, submarineName)
	err = operatorFramework.CreateSubmarine(framework.SubmarineClient, submarineNs, submarine)
	assert.Equal(t, nil, err)

	// wait for submarine to be in RUNNING state
	t.Logf("[Wait] Submarine %s/%s", submarineNs, submarineName)
	status := GetJobStatus(t, submarineNs, submarineName)
	err = wait.Poll(INTERVAL, TIMEOUT, func() (done bool, err error) {
		if status == v1alpha1.RunningState {
			return true, nil
		}
		status = GetJobStatus(t, submarineNs, submarineName)

		return false, nil
	})
	assert.Equal(t, nil, err)

	// delete submarine
	t.Logf("[Delete] Submarine %s/%s", submarineNs, submarineName)
	err = operatorFramework.DeleteSubmarine(framework.SubmarineClient, submarineNs, submarineName)
	assert.Equal(t, nil, err)

	// wait for submarine to be deleted entirely
	_, getError := operatorFramework.GetSubmarine(framework.SubmarineClient, submarineNs, submarineName)
	err = wait.Poll(INTERVAL, TIMEOUT, func() (done bool, err error) {
		if apierrors.IsNotFound(getError) {
			return true, nil
		}
		_, getError = operatorFramework.GetSubmarine(framework.SubmarineClient, submarineNs, submarineName)
		return false, nil
	})
	assert.Equal(t, nil, err)

	// delete the test namespace
	err = framework.KubeClient.CoreV1().Namespaces().Delete(context.TODO(), submarineNs, metav1.DeleteOptions{})
	assert.Equal(t, nil, err)
}
