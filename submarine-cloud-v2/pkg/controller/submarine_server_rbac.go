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

package controller

import (
	"context"
	"fmt"

	v1alpha1 "github.com/apache/submarine/submarine-cloud-v2/pkg/apis/submarine/v1alpha1"

	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
)

func newSubmarineServerClusterRole(submarine *v1alpha1.Submarine) *rbacv1.ClusterRole {
	return &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: serverName,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(submarine, v1alpha1.SchemeGroupVersion.WithKind("Submarine")),
			},
		},
		Rules: []rbacv1.PolicyRule{
			{
				Verbs:     []string{"get", "list", "watch", "create", "delete", "deletecollection", "patch", "update"},
				APIGroups: []string{"kubeflow.org"},
				Resources: []string{"tfjobs", "tfjobs/status", "pytorchjobs", "pytorchjobs/status", "notebooks", "notebooks/status"},
			},
			{
				Verbs:     []string{"get", "list", "watch", "create", "delete", "deletecollection", "patch", "update"},
				APIGroups: []string{"traefik.containo.us"},
				Resources: []string{"ingressroutes"},
			},
			{
				Verbs:     []string{"*"},
				APIGroups: []string{""},
				Resources: []string{"pods", "pods/log", "services", "persistentvolumes", "persistentvolumeclaims"},
			},
			{
				Verbs:     []string{"*"},
				APIGroups: []string{"apps"},
				Resources: []string{"deployments", "deployments/status"},
			},
		},
	}
}

func newSubmarineServerClusterRoleBinding(submarine *v1alpha1.Submarine) *rbacv1.ClusterRoleBinding {
	return &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: serverName,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(submarine, v1alpha1.SchemeGroupVersion.WithKind("Submarine")),
			},
		},
		Subjects: []rbacv1.Subject{
			{
				Kind:      "ServiceAccount",
				Namespace: submarine.Namespace,
				Name:      serverName,
			},
		},
		RoleRef: rbacv1.RoleRef{
			Kind:     "ClusterRole",
			Name:     serverName,
			APIGroup: "rbac.authorization.k8s.io",
		},
	}
}

// createSubmarineServerRBAC is a function to create RBAC for submarine-server.
// Reference: https://github.com/apache/submarine/blob/master/helm-charts/submarine/templates/rbac.yaml
func (c *Controller) createSubmarineServerRBAC(submarine *v1alpha1.Submarine) error {
	klog.Info("[createSubmarineServerRBAC]")

	// Step1: Create ClusterRole
	clusterrole, err := c.clusterroleLister.Get(serverName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		clusterrole, err = c.kubeclientset.RbacV1().ClusterRoles().Create(context.TODO(), newSubmarineServerClusterRole(submarine), metav1.CreateOptions{})
		klog.Info("	Create ClusterRole: ", clusterrole.Name)
	}

	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if err != nil {
		return err
	}

	if !metav1.IsControlledBy(clusterrole, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, clusterrole.Name)
		c.recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return fmt.Errorf(msg)
	}

	clusterrolebinding, clusterrolebinding_err := c.clusterrolebindingLister.Get(serverName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(clusterrolebinding_err) {
		clusterrolebinding, clusterrolebinding_err = c.kubeclientset.RbacV1().ClusterRoleBindings().Create(context.TODO(), newSubmarineServerClusterRoleBinding(submarine), metav1.CreateOptions{})
		klog.Info("	Create ClusterRoleBinding: ", clusterrolebinding.Name)
	}

	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if clusterrolebinding_err != nil {
		return clusterrolebinding_err
	}

	if !metav1.IsControlledBy(clusterrolebinding, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, clusterrolebinding.Name)
		c.recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return fmt.Errorf(msg)
	}

	return nil
}
