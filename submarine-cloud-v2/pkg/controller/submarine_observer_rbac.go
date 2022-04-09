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

func newSubmarineObserverRole(submarine *v1alpha1.Submarine) *rbacv1.Role {
	role, err := ParseRoleYaml(observerRbacYamlPath)
	if err != nil {
		klog.Info("[Error] ParseRole", err)
	}
	role.ObjectMeta.OwnerReferences = []metav1.OwnerReference{
		*metav1.NewControllerRef(submarine, v1alpha1.SchemeGroupVersion.WithKind("Submarine")),
	}

	return role
}

func newSubmarineObserverRoleBinding(submarine *v1alpha1.Submarine) *rbacv1.RoleBinding {
	roleBinding, err := ParseRoleBindingYaml(observerRbacYamlPath)
	if err != nil {
		klog.Info("[Error] ParseRoleBinding", err)
	}
	roleBinding.ObjectMeta.OwnerReferences = []metav1.OwnerReference{
		*metav1.NewControllerRef(submarine, v1alpha1.SchemeGroupVersion.WithKind("Submarine")),
	}

	return roleBinding
}

// createSubmarineObserverRBAC is a function to create RBAC for submarine-observer which will be binded on service account: default.
// Reference: https://github.com/apache/submarine/blob/master/helm-charts/submarine/templates/rbac.yaml
func (c *Controller) createSubmarineObserverRBAC(submarine *v1alpha1.Submarine) error {
	klog.Info("[createSubmarineObserverRBAC]")

	// Step1: Create Role
	role, err := c.roleLister.Roles(submarine.Namespace).Get(observerName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		role, err = c.kubeclientset.RbacV1().Roles(submarine.Namespace).Create(context.TODO(), newSubmarineObserverRole(submarine), metav1.CreateOptions{})
		klog.Info("	Create Role: ", role.Name)
	}

	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if err != nil {
		return err
	}

	if !metav1.IsControlledBy(role, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, role.Name)
		c.recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return fmt.Errorf(msg)
	}

	rolebinding, rolebinding_err := c.rolebindingLister.RoleBindings(submarine.Namespace).Get(observerName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(rolebinding_err) {
		rolebinding, rolebinding_err = c.kubeclientset.RbacV1().RoleBindings(submarine.Namespace).Create(context.TODO(), newSubmarineObserverRoleBinding(submarine), metav1.CreateOptions{})
		klog.Info("	Create RoleBinding: ", rolebinding.Name)
	}

	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if rolebinding_err != nil {
		return rolebinding_err
	}

	if !metav1.IsControlledBy(rolebinding, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, rolebinding.Name)
		c.recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return fmt.Errorf(msg)
	}

	return nil
}
