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
	"fmt"
	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"os"
	"path/filepath"
	ctrl "sigs.k8s.io/controller-runtime"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/pkg/errors"
	istio "istio.io/client-go/pkg/apis/networking/v1alpha3"
	istiov1alpha3 "istio.io/client-go/pkg/apis/networking/v1alpha3"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/yaml"
	"sigs.k8s.io/controller-runtime/pkg/client"

	submarineapacheorgv1 "github.com/apache/submarine/submarine-cloud-v3/api/v1"
)

func createScheme() *runtime.Scheme {
	scheme := runtime.NewScheme()
	_ = clientgoscheme.AddToScheme(scheme)
	_ = appsv1.AddToScheme(scheme)
	_ = corev1.AddToScheme(scheme)
	_ = submarineapacheorgv1.AddToScheme(scheme)
	_ = istio.AddToScheme(scheme)
	_ = serializer.NewCodecFactory(scheme).UniversalDeserializer().Decode
	return scheme
}

// We are mainly testing created resources, so for the time being we have not declared k8s client and recoder
func createSubmarineReconciler(config ...*SubmarineReconciler) *SubmarineReconciler {
	namespace := "submarine"
	istioEnable := false
	submarineGateway := "submarine/submarine-gateway"
	seldonIstioEnable := false
	seldonGateway := "submarine/seldon-gateway"
	clusterType := "kubernetes"
	createPodSecurityPolicy := false
	if len(config) > 0 {
		if config[0].Namespace != "" {
			namespace = config[0].Namespace
		}
		if &config[0].IstioEnable != nil {
			istioEnable = config[0].IstioEnable
		}
		if config[0].SubmarineGateway != "" {
			submarineGateway = config[0].SubmarineGateway
		}
		if &config[0].SeldonIstioEnable != nil {
			seldonIstioEnable = config[0].SeldonIstioEnable
		}
		if config[0].SeldonGateway != "" {
			seldonGateway = config[0].SeldonGateway
		}
		if config[0].ClusterType != "" {
			clusterType = config[0].ClusterType
		}
		if &config[0].CreatePodSecurityPolicy != nil {
			createPodSecurityPolicy = config[0].CreatePodSecurityPolicy
		}
	}
	return &SubmarineReconciler{
		Scheme:                  createScheme(),
		Log:                     ctrl.Log.WithName("submarine-test"),
		Namespace:               namespace,
		IstioEnable:             istioEnable,
		SubmarineGateway:        submarineGateway,
		SeldonIstioEnable:       seldonIstioEnable,
		SeldonGateway:           seldonGateway,
		ClusterType:             clusterType,
		CreatePodSecurityPolicy: createPodSecurityPolicy,
	}
}

var _ = Describe("Submarine controller", func() {

	// Define utility constants and variables
	const (
		// The namespaces where the Submarine CRs are created
		submarineNamespaceDefaultCR = "submarine-test-submit-default-cr"
		submarineNamespaceCustomCR  = "submarine-test-submit-custom-cr"
		customHost                  = "submarine-custom-host"
		customGateway               = "submarine-custom-gateway"

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
		submarineNameDefaultCR  string
		submarineNameCustomCR   string
		submarineCustomHosts    []string
		submarineCustomGateways []string

		ctx = context.Background()
	)

	Context("Create test namespaces", func() {
		It(fmt.Sprintf("Should create namespace %s", submarineNamespaceDefaultCR), func() {
			By(fmt.Sprintf("Creating the test namespace %s", submarineNamespaceDefaultCR))
			ns := &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: submarineNamespaceDefaultCR, // Namespace to test default CR
					Labels: map[string]string{
						"istio-injection": "enabled",
					},
				},
			}
			Expect(k8sClient.Create(ctx, ns)).Should(Succeed())

			// We'll need to retry getting this newly created namespace, given that creation may not immediately happen.
			createdNs := &corev1.Namespace{} // stub
			Eventually(func() bool {
				err := k8sClient.Get(ctx, types.NamespacedName{Name: submarineNamespaceDefaultCR, Namespace: "default"}, createdNs)
				if err != nil {
					return false
				}
				return true
			}, createNsTimeout, createNsInterval).Should(BeTrue())

			// The namespace should have Istio label
			Expect(createdNs.Labels["istio-injection"]).To(Equal("enabled"))
		})
		It(fmt.Sprintf("Should create namespace %s", submarineNamespaceCustomCR), func() {
			By(fmt.Sprintf("Creating the test namespace %s", submarineNamespaceCustomCR))
			ns := &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: submarineNamespaceCustomCR, // Namespace to test custom CR
					Labels: map[string]string{
						"istio-injection": "enabled",
					},
				},
			}
			Expect(k8sClient.Create(ctx, ns)).Should(Succeed())

			// We'll need to retry getting this newly created namespace, given that creation may not immediately happen.
			createdNs := &corev1.Namespace{} // stub
			Eventually(func() bool {
				err := k8sClient.Get(ctx, types.NamespacedName{Name: submarineNamespaceCustomCR, Namespace: "default"}, createdNs)
				if err != nil {
					return false
				}
				return true
			}, createNsTimeout, createNsInterval).Should(BeTrue())

			// The namespace should have Istio label
			Expect(createdNs.Labels["istio-injection"]).To(Equal("enabled"))
		})
	})

	Context("Create Submarines", func() {
		It(fmt.Sprintf("Should create Submarine in %s and it should become RUNNING", submarineNamespaceDefaultCR), func() {
			By(fmt.Sprintf("Creating new Submarine in %s", submarineNamespaceDefaultCR))
			submarine, err := MakeSubmarineFromYaml("../config/samples/_v1_submarine.yaml")
			Expect(err).To(BeNil())

			// Leave Spec.Virtualservice.Host empty to test default value
			// Leave Spec.Virtualservice.Gateways empty to test default value

			// The name of Submarine is specified in the YAML file.
			// Storing name to call k8sClient.Get with NamespacedName
			submarineNameDefaultCR = submarine.Name

			// Create Submarines in our namespace
			submarine.Namespace = submarineNamespaceDefaultCR
			Expect(k8sClient.Create(ctx, submarine)).Should(Succeed())

			// We'll need to retry getting this newly created Submarine, given that creation may not immediately happen.
			createdSubmarine := &submarineapacheorgv1.Submarine{} // stub
			Eventually(func() bool {
				err := k8sClient.Get(ctx, types.NamespacedName{Name: submarineNameDefaultCR, Namespace: submarineNamespaceDefaultCR}, createdSubmarine)
				if err != nil {
					return false
				}
				return true
			}, createNsTimeout, createNsInterval).Should(BeTrue())

			// Wait for Submarine to be in RUNNING state
			By(fmt.Sprintf("Waiting until Submarine %s/%s become RUNNING", submarineNameDefaultCR, submarineNamespaceDefaultCR))
			Eventually(func() bool {
				err = k8sClient.Get(ctx, types.NamespacedName{Name: submarineNameDefaultCR, Namespace: submarineNamespaceDefaultCR}, createdSubmarine)
				Expect(err).To(BeNil())

				state := createdSubmarine.Status.SubmarineState.State
				Expect(state).ToNot(Equal(submarineapacheorgv1.FailedState))
				if createdSubmarine.Status.SubmarineState.State == submarineapacheorgv1.RunningState {
					return true
				}
				return false
			}, createSubmarineTimeout, createSubmarineInterval).Should(BeTrue())
		})
		It(fmt.Sprintf("Should create Submarine in %s and it should become RUNNING", submarineNamespaceCustomCR), func() {
			By(fmt.Sprintf("Creating new Submarine in %s", submarineNamespaceCustomCR))
			submarine, err := MakeSubmarineFromYaml("../config/samples/_v1_submarine.yaml")
			Expect(err).To(BeNil())

			// Set Spec.Virtualservice.Hosts to [submarineCustomHosts] to test custom value
			submarineCustomHosts = make([]string, 1, 1)
			submarineCustomHosts[0] = customHost
			submarine.Spec.Virtualservice.Hosts = submarineCustomHosts

			// Set Spec.Virtualservice.Gateways to [submarineCustomGateway] to test custom value
			submarineCustomGateways = make([]string, 1, 1)
			submarineCustomGateways[0] = customGateway
			submarine.Spec.Virtualservice.Gateways = submarineCustomGateways

			// The name of Submarine is specified in the YAML file.
			// Storing name to call k8sClient.Get with NamespacedName
			submarineNameCustomCR = submarine.Name

			// Create Submarines in our namespace
			submarine.Namespace = submarineNamespaceCustomCR
			Expect(k8sClient.Create(ctx, submarine)).Should(Succeed())

			// We'll need to retry getting this newly created Submarine, given that creation may not immediately happen.
			createdSubmarine := &submarineapacheorgv1.Submarine{} // stub
			Eventually(func() bool {
				err := k8sClient.Get(ctx, types.NamespacedName{Name: submarineNameCustomCR, Namespace: submarineNamespaceCustomCR}, createdSubmarine)
				if err != nil {
					return false
				}
				return true
			}, createNsTimeout, createNsInterval).Should(BeTrue())

			// Wait for Submarine to be in RUNNING state
			By(fmt.Sprintf("Waiting until Submarine %s/%s become RUNNING", submarineNameCustomCR, submarineNamespaceCustomCR))
			Eventually(func() bool {
				err = k8sClient.Get(ctx, types.NamespacedName{Name: submarineNameCustomCR, Namespace: submarineNamespaceCustomCR}, createdSubmarine)
				Expect(err).To(BeNil())

				state := createdSubmarine.Status.SubmarineState.State
				Expect(state).ToNot(Equal(submarineapacheorgv1.FailedState))
				if createdSubmarine.Status.SubmarineState.State == submarineapacheorgv1.RunningState {
					return true
				}
				return false
			}, createSubmarineTimeout, createSubmarineInterval).Should(BeTrue())
		})
	})

	Context("Verify Virtual Service Spec", func() {
		It(fmt.Sprintf("Hosts and Gateways should have default values In %s", submarineNamespaceDefaultCR), func() {
			By(fmt.Sprintf("Getting Virtual Service In %s", submarineNamespaceDefaultCR))
			createdVirtualService := &istiov1alpha3.VirtualService{} // stub
			err := k8sClient.Get(ctx, types.NamespacedName{Name: virtualServiceName, Namespace: submarineNamespaceDefaultCR}, createdVirtualService)
			Expect(err).To(BeNil())

			// The default value for host is *
			Expect(createdVirtualService.Spec.Hosts[0]).To(Equal("*"))
			// The default value for gateway is ${namespace}/submarine-gateway
			Expect(createdVirtualService.Spec.Gateways[0]).To(Equal("submarine-cloud-v3-system/submarine-gateway"))
		})
		It(fmt.Sprintf("Hosts and Gateways should have custom values In %s", submarineNamespaceCustomCR), func() {
			By(fmt.Sprintf("Getting Virtual Service In %s", submarineNamespaceCustomCR))
			createdVirtualService := &istiov1alpha3.VirtualService{} // stub
			err := k8sClient.Get(ctx, types.NamespacedName{Name: virtualServiceName, Namespace: submarineNamespaceCustomCR}, createdVirtualService)
			Expect(err).To(BeNil())

			// The custom value for hosts matches the submarine CR
			Expect(createdVirtualService.Spec.Hosts).To(Equal(submarineCustomHosts))
			// The custom value for gateways matches the submarine CR
			Expect(createdVirtualService.Spec.Gateways).To(Equal(submarineCustomGateways))
		})
	})

	Context("Delete Submarine", func() {
		It(fmt.Sprintf("Should delete the Submarine In %s", submarineNamespaceDefaultCR), func() {
			Expect(submarineNameDefaultCR).ToNot(BeNil())

			By(fmt.Sprintf("Deleting Submarine %s/%s", submarineNameDefaultCR, submarineNamespaceDefaultCR))
			createdSubmarine := &submarineapacheorgv1.Submarine{} // stub
			err := k8sClient.Get(ctx, types.NamespacedName{Name: submarineNameDefaultCR, Namespace: submarineNamespaceDefaultCR}, createdSubmarine)
			Expect(err).To(BeNil())

			foreground := metav1.DeletePropagationForeground
			err = k8sClient.Delete(ctx, createdSubmarine, &client.DeleteOptions{
				PropagationPolicy: &foreground,
			})
			Expect(err).To(BeNil())

			// Wait for Submarine to be deleted entirely
			Eventually(func() bool {
				err := k8sClient.Get(ctx, types.NamespacedName{Name: submarineNameDefaultCR, Namespace: submarineNamespaceDefaultCR}, createdSubmarine)
				if apierrors.IsNotFound(err) {
					return true
				}
				Expect(err).To(BeNil())
				return false
			}, deleteSubmarineTimeout, deleteSubmarineInterval).Should(BeTrue())
		})
		It(fmt.Sprintf("Should delete the Submarine In %s", submarineNamespaceCustomCR), func() {
			Expect(submarineNameCustomCR).ToNot(BeNil())

			By(fmt.Sprintf("Deleting Submarine %s/%s", submarineNameCustomCR, submarineNamespaceCustomCR))
			createdSubmarine := &submarineapacheorgv1.Submarine{} // stub
			err := k8sClient.Get(ctx, types.NamespacedName{Name: submarineNameCustomCR, Namespace: submarineNamespaceCustomCR}, createdSubmarine)
			Expect(err).To(BeNil())

			foreground := metav1.DeletePropagationForeground
			err = k8sClient.Delete(ctx, createdSubmarine, &client.DeleteOptions{
				PropagationPolicy: &foreground,
			})
			Expect(err).To(BeNil())

			// Wait for Submarine to be deleted entirely
			Eventually(func() bool {
				err := k8sClient.Get(ctx, types.NamespacedName{Name: submarineNameCustomCR, Namespace: submarineNamespaceCustomCR}, createdSubmarine)
				if apierrors.IsNotFound(err) {
					return true
				}
				Expect(err).To(BeNil())
				return false
			}, deleteSubmarineTimeout, deleteSubmarineInterval).Should(BeTrue())
		})
	})

	Context("Delete the test namespace", func() {
		It(fmt.Sprintf("Should delete namespace %s", submarineNamespaceDefaultCR), func() {
			By(fmt.Sprintf("Deleting the test namespace %s", submarineNamespaceDefaultCR))

			createdNs := &corev1.Namespace{} // stub
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: submarineNamespaceDefaultCR, Namespace: "default"}, createdNs)).Should(Succeed())
			Expect(k8sClient.Delete(ctx, createdNs)).Should(Succeed())

			// Wait for submarine to be deleted entirely

			Eventually(func() bool {
				err := k8sClient.Get(ctx, types.NamespacedName{Name: submarineNamespaceDefaultCR, Namespace: "default"}, createdNs)
				if apierrors.IsNotFound(err) {
					return true
				}
				Expect(err).To(BeNil())
				return false
			}, deleteNsTimeout, deleteNsInterval).Should(BeTrue())
		})
		It(fmt.Sprintf("Should delete namespace %s", submarineNamespaceCustomCR), func() {
			By(fmt.Sprintf("Deleting the test namespace %s", submarineNamespaceCustomCR))

			createdNs := &corev1.Namespace{} // stub
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: submarineNamespaceCustomCR, Namespace: "default"}, createdNs)).Should(Succeed())
			Expect(k8sClient.Delete(ctx, createdNs)).Should(Succeed())

			// Wait for submarine to be deleted entirely

			Eventually(func() bool {
				err := k8sClient.Get(ctx, types.NamespacedName{Name: submarineNamespaceCustomCR, Namespace: "default"}, createdNs)
				if apierrors.IsNotFound(err) {
					return true
				}
				Expect(err).To(BeNil())
				return false
			}, deleteNsTimeout, deleteNsInterval).Should(BeTrue())
		})
	})
})

func MakeSubmarineFromYaml(pathToYaml string) (*submarineapacheorgv1.Submarine, error) {
	manifest, err := PathToOSFile(pathToYaml)
	if err != nil {
		return nil, err
	}
	tmp := submarineapacheorgv1.Submarine{}
	if err := yaml.NewYAMLOrJSONDecoder(manifest, 100).Decode(&tmp); err != nil {
		return nil, errors.Wrap(err, fmt.Sprintf("failed to decode file %s", pathToYaml))
	}
	return &tmp, err
}

func MakeSubmarineFromYamlByNamespace(pathToYaml string, namespace string) (*submarineapacheorgv1.Submarine, error) {
	submarine, err := MakeSubmarineFromYaml(pathToYaml)
	if err != nil {
		return nil, err
	}
	submarine.Namespace = namespace
	return submarine, nil
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
