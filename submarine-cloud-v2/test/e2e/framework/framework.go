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

package framework

import (
	"context"
	"time"

	clientset "submarine-cloud-v2/pkg/generated/clientset/versioned"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
)

type Framework struct {
	KubeClient	kubernetes.Interface
	SubmarineClient clientset.Interface
	Namespace *corev1.Namespace
	OperatorPod *corev1.Pod
	MasterHost string
	DefaultTimeout time.Duration
}

var SubmarineTestNamespace = "submarine-user-test"


func New(ns, submarineNs, kubeconfig, opImage, opImagePullPolicy string) (*Framework, error) {
	
	cfg, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
	if err != nil {
		return nil, errors.Wrap(err, "build config failed")
	}

	kubeClient, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		return nil, errors.Wrap(err, "creating new kube-client fail")
	}
	
	// create submarine-operator namespace
	namespace, err := kubeClient.CoreV1().Namespaces().Create(context.TODO(), &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: ns,
		},
	},
	metav1.CreateOptions{})
	if apierrors.IsAlreadyExists(err) {
		namespace, err = kubeClient.CoreV1().Namespaces().Get(context.TODO(), ns, metav1.GetOptions{})
	} else {
		return nil, errors.Wrap(err, "create submarine operator namespace fail")
	}

	submarineClient, err := clientset.NewForConfig(cfg)
	if err != nil {
		return nil, errors.Wrap(err, "creating new submarine-client fail")
	}

	f := &Framework {
		MasterHost: cfg.Host,
		KubeClient: kubeClient,
		SubmarineClient: submarineClient,
		Namespace: namespace,
		DefaultTimeout: time.Minute,
	}
	err = f.Setup(submarineNs, opImage, opImagePullPolicy)
	if err != nil {
		return nil, errors.Wrap(err, "setup test environment failed")
	}

	return f, nil
}

func (f *Framework) Setup(submarineNs, opImage, opImagePullPolicy string) error {
	if err := f.setupOperator(submarineNs, opImage, opImagePullPolicy); err != nil {
		return errors.Wrap(err, "setup operator failed")
	}
	return nil
}

func (f* Framework) setupOperator(submarineNs, opImage, opImagePullPolicy string) error {

	// setup RBAC (ClusterRole, ClusterRoleBinding, and ServiceAccount)
	if _, err := CreateServiceAccount(f.KubeClient, f.Namespace.Name, "../../artifacts/examples/submarine-operator-service-account.yaml"); err != nil && !apierrors.IsAlreadyExists(err) {
		return errors.Wrap(err, "failed to create operator service account")
	}

	if err := CreateClusterRole(f.KubeClient, "../../artifacts/examples/submarine-operator-service-account.yaml"); err != nil && !apierrors.IsAlreadyExists(err) {
		return errors.Wrap(err, "failed to create cluster role")
	}

	if _, err := CreateClusterRoleBinding(f.KubeClient, "../../artifacts/examples/submarine-operator-service-account.yaml"); err != nil && !apierrors.IsAlreadyExists(err) {
		return errors.Wrap(err, "failed to create cluster role binding")
	}
	// Deploy a submarine-operator
	deploy, err := MakeDeployment("../../artifacts/examples/submarine-operator.yaml")
	if err != nil {
		return err
	}

	if opImage != "" {
		// Override operator image used, if specified when running tests.
		deploy.Spec.Template.Spec.Containers[0].Image = opImage
	}

	for _, container := range deploy.Spec.Template.Spec.Containers {
		container.ImagePullPolicy = corev1.PullPolicy(opImagePullPolicy)
	}

	err = CreateDeployment(f.KubeClient, f.Namespace.Name, deploy)
	if err != nil {
		return err
	}

	opts := metav1.ListOptions{LabelSelector: fields.SelectorFromSet(fields.Set(deploy.Spec.Template.ObjectMeta.Labels)).String()}
	err = WaitForPodsReady(f.KubeClient, f.Namespace.Name, f.DefaultTimeout, 1, opts)
	if err != nil {
		return errors.Wrap(err, "failed to wait for operator to become ready")
	}

	pl, err := f.KubeClient.CoreV1().Pods(f.Namespace.Name).List(context.TODO(), opts)
	if err != nil {
		return err
	}
	f.OperatorPod = &pl.Items[0]

	return nil
}

// Teardown ters down a previously initialized test environment
func (f *Framework) Teardown() error {
	if err := DeleteClusterRole(f.KubeClient, "../../artifacts/examples/submarine-operator-service-account.yaml"); err != nil && !apierrors.IsAlreadyExists(err) {
		return errors.Wrap(err, "failed to delete operator cluster role")
	}

	if err := DeleteClusterRoleBinding(f.KubeClient, "../../artifacts/examples/submarine-operator-service-account.yaml"); err != nil && !apierrors.IsAlreadyExists(err) {
		return errors.Wrap(err, "failed to delete operator cluster role binding")
	}

	if err := f.KubeClient.AppsV1().Deployments(f.Namespace.Name).Delete(context.TODO(), "submarine-operator-demo", metav1.DeleteOptions{}); err != nil {
		return err
	}

	if err := DeleteNamespace(f.KubeClient, f.Namespace.Name); err != nil && !apierrors.IsForbidden(err) {
		return err
	}

	return nil
}