/*
Copyright 2022.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controllers

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/yaml"
	"sigs.k8s.io/controller-runtime/pkg/client"

	submarineapacheorgv1alpha1 "github.com/apache/submarine/submarine-cloud-v3/api/v1alpha1"
)

var _ = Describe("Submarine controller", func() {

	// Define utility constants and variables
	const (
		submarineNamespace = "submarine-test-submit-custom-resource" // The namespace where the Submarine CR is created

		createNsTimeout         = time.Second * 10
		createNsInterval        = time.Second * 2
		createSubmarineTimeout  = time.Second * 1200
		createSubmarineInterval = time.Second * 10
		deleteSubmarineTimeout  = time.Second * 600
		deleteSubmarineInterval = time.Second * 10
		deleteNsTimeout         = time.Second * 120
		deleteNsInterval        = time.Second * 2
	)
	var (
		// The name of Submarine is specified in the YAML file.
		// Storing name to call k8sClient.Get with NamespacedName
		submarineName string

		ctx = context.Background()
	)

	Context("Create a test namespace", func() {
		It("Should create a test namespace", func() {
			msg := fmt.Sprintf("Creating the test namespace %s", submarineNamespace)
			By(msg)

			ns := &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: submarineNamespace,
					Labels: map[string]string{
						"istio-injection": "enabled",
					},
				},
			}
			Expect(k8sClient.Create(ctx, ns)).Should(Succeed())

			// We'll need to retry getting this newly created namespace, given that creation may not immediately happen.
			createdNs := &corev1.Namespace{} // stub
			Eventually(func() bool {
				err := k8sClient.Get(ctx, types.NamespacedName{Name: submarineNamespace, Namespace: "default"}, createdNs)
				if err != nil {
					return false
				}
				return true
			}, createNsTimeout, createNsInterval).Should(BeTrue())

			// The namespace should have Istio label
			Expect(createdNs.Labels["istio-injection"]).To(Equal("enabled"))
		})
	})

	Context("Create Submarine", func() {
		It("Should create a Submarine and it should become RUNNING", func() {
			By("Creating a new Submarine")
			submarine, err := MakeSubmarineFromYaml("../config/samples/_v1alpha1_submarine.yaml")
			Expect(err).To(BeNil())

			// The name of Submarine is specified in the YAML file.
			// Storing name to call k8sClient.Get with NamespacedName
			submarineName = submarine.Name

			// Create Submarine in our namespace
			submarine.Namespace = submarineNamespace
			Expect(k8sClient.Create(ctx, submarine)).Should(Succeed())

			// We'll need to retry getting this newly created Submarine, given that creation may not immediately happen.
			createdSubmarine := &submarineapacheorgv1alpha1.Submarine{} // stub
			Eventually(func() bool {
				err := k8sClient.Get(ctx, types.NamespacedName{Name: submarineName, Namespace: submarineNamespace}, createdSubmarine)
				if err != nil {
					return false
				}
				return true
			}, createNsTimeout, createNsInterval).Should(BeTrue())

			// Wait for Submarine to be in RUNNING state
			msg := fmt.Sprintf("Waiting until Submarine %s/%s become RUNNING", submarineName, submarineNamespace)
			By(msg)
			Eventually(func() bool {
				err = k8sClient.Get(ctx, types.NamespacedName{Name: submarineName, Namespace: submarineNamespace}, createdSubmarine)
				Expect(err).To(BeNil())

				state := createdSubmarine.Status.SubmarineState.State
				Expect(state).ToNot(Equal(submarineapacheorgv1alpha1.FailedState))
				if createdSubmarine.Status.SubmarineState.State == submarineapacheorgv1alpha1.RunningState {
					return true
				}
				return false
			}, createSubmarineTimeout, createSubmarineInterval).Should(BeTrue())
		})
	})

	Context("Delete Submarine", func() {
		It("Should delete the Submarine", func() {
			Expect(submarineName).ToNot(BeNil())

			msg := fmt.Sprintf("Deleting Submarine %s/%s", submarineName, submarineNamespace)
			By(msg)

			createdSubmarine := &submarineapacheorgv1alpha1.Submarine{} // stub
			err := k8sClient.Get(ctx, types.NamespacedName{Name: submarineName, Namespace: submarineNamespace}, createdSubmarine)
			Expect(err).To(BeNil())

			foreground := metav1.DeletePropagationForeground
			err = k8sClient.Delete(ctx, createdSubmarine, &client.DeleteOptions{
				PropagationPolicy: &foreground,
			})
			Expect(err).To(BeNil())

			// Wait for Submarine to be deleted entirely
			Eventually(func() bool {
				err := k8sClient.Get(ctx, types.NamespacedName{Name: submarineName, Namespace: submarineNamespace}, createdSubmarine)
				if apierrors.IsNotFound(err) {
					return true
				}
				Expect(err).To(BeNil())
				return false
			}, deleteSubmarineTimeout, deleteSubmarineInterval).Should(BeTrue())
		})
	})

	Context("Delete the test namespace", func() {
		It("Should delete the test namespace", func() {
			msg := fmt.Sprintf("Deleting the test namespace %s", submarineNamespace)
			By(msg)

			createdNs := &corev1.Namespace{} // stub
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: submarineNamespace, Namespace: "default"}, createdNs)).Should(Succeed())
			Expect(k8sClient.Delete(ctx, createdNs)).Should(Succeed())

			// Wait for submarine to be deleted entirely

			Eventually(func() bool {
				err := k8sClient.Get(ctx, types.NamespacedName{Name: submarineNamespace, Namespace: "default"}, createdNs)
				if apierrors.IsNotFound(err) {
					return true
				}
				Expect(err).To(BeNil())
				return false
			}, deleteNsTimeout, deleteNsInterval).Should(BeTrue())
		})
	})
})

func MakeSubmarineFromYaml(pathToYaml string) (*submarineapacheorgv1alpha1.Submarine, error) {
	manifest, err := PathToOSFile(pathToYaml)
	if err != nil {
		return nil, err
	}
	tmp := submarineapacheorgv1alpha1.Submarine{}
	if err := yaml.NewYAMLOrJSONDecoder(manifest, 100).Decode(&tmp); err != nil {
		return nil, errors.Wrap(err, fmt.Sprintf("failed to decode file %s", pathToYaml))
	}
	return &tmp, err
}

// PathToOSFile gets the absolute path from relative path.
func PathToOSFile(relativePath string) (*os.File, error) {
	path, err := filepath.Abs(relativePath)
	if err != nil {
		return nil, errors.Wrap(err, fmt.Sprintf("failed generate absolute file path of %s", relativePath))
	}

	manifest, err := os.Open(path)
	if err != nil {
		return nil, errors.Wrap(err, fmt.Sprintf("failed to open file %s", path))
	}

	return manifest, nil
}
