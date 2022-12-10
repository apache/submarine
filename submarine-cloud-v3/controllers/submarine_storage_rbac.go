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
	"github.com/apache/submarine/submarine-cloud-v3/controllers/util"

	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	submarineapacheorgv1alpha1 "github.com/apache/submarine/submarine-cloud-v3/api/v1alpha1"

	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

func (r *SubmarineReconciler) newSubmarineStorageRole(ctx context.Context, submarine *submarineapacheorgv1alpha1.Submarine) *rbacv1.Role {
	role, err := util.ParseRoleYaml(storageRbacYamlPath)
	if err != nil {
		r.Log.Error(err, "ParseRoleYaml")
	}
	role.Namespace = submarine.Namespace
	err = controllerutil.SetControllerReference(submarine, role, r.Scheme)
	if err != nil {
		r.Log.Error(err, "Set Role ControllerReference")
	}

	// If cluster type is openshift and need create pod security policy, we need add anyuid scc, or we add k8s psp
	if r.ClusterType == "openshift" {
		role.Rules = append(role.Rules, openshiftAnyuidRoleRule)
	} else {
		role.Rules = append(role.Rules, k8sAnyuidRoleRule)
	}

	return role
}

func (r *SubmarineReconciler) newSubmarineStorageRoleBinding(ctx context.Context, submarine *submarineapacheorgv1alpha1.Submarine) *rbacv1.RoleBinding {
	roleBinding, err := util.ParseRoleBindingYaml(storageRbacYamlPath)
	if err != nil {
		r.Log.Error(err, "Set RoleBinding ControllerReference")
	}
	roleBinding.Namespace = submarine.Namespace
	err = controllerutil.SetControllerReference(submarine, roleBinding, r.Scheme)
	if err != nil {
		r.Log.Error(err, "Set RoleBinding ControllerReference")
	}
	return roleBinding
}

func (r *SubmarineReconciler) newSubmarineStorageServiceAccount(ctx context.Context, submarine *submarineapacheorgv1alpha1.Submarine) *corev1.ServiceAccount {
	serviceAccount, err := util.ParseServiceAccountYaml(storageRbacYamlPath)
	if err != nil {
		r.Log.Error(err, "ParseServiceAccountYaml")
	}
	serviceAccount.Namespace = submarine.Namespace
	err = controllerutil.SetControllerReference(submarine, serviceAccount, r.Scheme)
	if err != nil {
		r.Log.Error(err, "Set ServiceAccount ControllerReference")
	}
	return serviceAccount
}

// createSubmarineStorageRBAC is a function to create RBAC for submarine-database and submarine-minio which will be binded on service account: submarine-storage.
// Reference: https://github.com/apache/submarine/blob/master/submarine-cloud-v3/artifacts/submarine-storage-rbac.yaml
func (r *SubmarineReconciler) createSubmarineStorageRBAC(ctx context.Context, submarine *submarineapacheorgv1alpha1.Submarine) error {
	r.Log.Info("Enter createSubmarineStorageRBAC")

	// Step1: Create ServiceAccount
	serviceaccount := &corev1.ServiceAccount{}
	err := r.Get(ctx, types.NamespacedName{Name: storageName, Namespace: submarine.Namespace}, serviceaccount)

	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		serviceaccount = r.newSubmarineStorageServiceAccount(ctx, submarine)
		err = r.Create(ctx, serviceaccount)
		r.Log.Info("Create ServiceAccount", "name", serviceaccount.Name)
	}

	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if err != nil {
		return err
	}

	// Step2: Pod Security Policy if needed
	if r.CreatePodSecurityPolicy {
		// Step2.1: Create Role
		role := &rbacv1.Role{}
		err = r.Get(ctx, types.NamespacedName{Name: storageName, Namespace: submarine.Namespace}, role)
		// If the resource doesn't exist, we'll create it
		if errors.IsNotFound(err) {
			role = r.newSubmarineStorageRole(ctx, submarine)
			err = r.Create(ctx, role)
			r.Log.Info("Create Role", "name", role.Name)
		}

		// If an error occurs during Get/Create, we'll requeue the item so we can
		// attempt processing again later. This could have been caused by a
		// temporary network failure, or any other transient reason.
		if err != nil {
			return err
		}

		if !metav1.IsControlledBy(role, submarine) {
			msg := fmt.Sprintf(MessageResourceExists, role.Name)
			r.Recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
			return fmt.Errorf(msg)
		}

		// Step2.2: Create Role Binding
		rolebinding := &rbacv1.RoleBinding{}
		err = r.Get(ctx, types.NamespacedName{Name: storageName, Namespace: submarine.Namespace}, rolebinding)
		// If the resource doesn't exist, we'll create it
		if errors.IsNotFound(err) {
			rolebinding = r.newSubmarineStorageRoleBinding(ctx, submarine)
			err = r.Create(ctx, rolebinding)
			r.Log.Info("Create RoleBinding", "name", rolebinding.Name)
		}

		// If an error occurs during Get/Create, we'll requeue the item so we can
		// attempt processing again later. This could have been caused by a
		// temporary network failure, or any other transient reason.
		if err != nil {
			return err
		}

		if !metav1.IsControlledBy(rolebinding, submarine) {
			msg := fmt.Sprintf(MessageResourceExists, rolebinding.Name)
			r.Recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
			return fmt.Errorf(msg)
		}
	}

	return nil
}
