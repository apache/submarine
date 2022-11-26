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

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	submarineapacheorgv1alpha1 "github.com/apache/submarine/submarine-cloud-v3/api/v1alpha1"

	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

func (r *SubmarineReconciler) newSubmarineDatabasePersistentVolumeClaim(ctx context.Context, submarine *submarineapacheorgv1alpha1.Submarine) *corev1.PersistentVolumeClaim {
	pvc, err := util.ParsePersistentVolumeClaimYaml(databaseYamlPath)
	if err != nil {
		r.Log.Error(err, "ParsePersistentVolumeClaimYaml")
	}
	pvc.Namespace = submarine.Namespace
	err = controllerutil.SetControllerReference(submarine, pvc, r.Scheme)
	if err != nil {
		r.Log.Error(err, "Set PVC ControllerReference")
	}
	return pvc
}

func (r *SubmarineReconciler) newSubmarineDatabaseStatefulSet(ctx context.Context, submarine *submarineapacheorgv1alpha1.Submarine) *appsv1.StatefulSet {
	statefulset, err := util.ParseStatefulSetYaml(databaseYamlPath)
	if err != nil {
		r.Log.Error(err, "ParseStatefulSetYaml")
	}

	statefulset.Namespace = submarine.Namespace
	err = controllerutil.SetControllerReference(submarine, statefulset, r.Scheme)
	if err != nil {
		r.Log.Error(err, "Set Stateful Set ControllerReference")
	}

	// database image
	databaseImage := submarine.Spec.Database.Image
	if databaseImage != "" {
		statefulset.Spec.Template.Spec.Containers[0].Image = databaseImage
	} else {
		statefulset.Spec.Template.Spec.Containers[0].Image = fmt.Sprintf("apache/submarine:database-%s", submarine.Spec.Version)
	}
	// pull secrets
	pullSecrets := util.GetSubmarineCommonImage(submarine).PullSecrets
	if pullSecrets != nil {
		statefulset.Spec.Template.Spec.ImagePullSecrets = r.CreatePullSecrets(&pullSecrets)
	}

	return statefulset
}

func (r *SubmarineReconciler) newSubmarineDatabaseService(ctx context.Context, submarine *submarineapacheorgv1alpha1.Submarine) *corev1.Service {
	service, err := util.ParseServiceYaml(databaseYamlPath)
	if err != nil {
		r.Log.Error(err, "ParseServiceYaml")
	}
	service.Namespace = submarine.Namespace
	err = controllerutil.SetControllerReference(submarine, service, r.Scheme)
	if err != nil {
		r.Log.Error(err, "Set Service ControllerReference")
	}
	return service
}

// createSubmarineDatabase is a function to create submarine-database.
// Reference: https://github.com/apache/submarine/blob/master/submarine-cloud-v3/artifacts/submarine-database.yaml
func (r *SubmarineReconciler) createSubmarineDatabase(ctx context.Context, submarine *submarineapacheorgv1alpha1.Submarine) error {
	r.Log.Info("Enter createSubmarineDatabase")

	// Step 1: Create PersistentVolumeClaim
	pvc := &corev1.PersistentVolumeClaim{}
	err := r.Get(ctx, types.NamespacedName{Name: databasePvcName, Namespace: submarine.Namespace}, pvc)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		pvc = r.newSubmarineDatabasePersistentVolumeClaim(ctx, submarine)
		err = r.Create(ctx, pvc)
		r.Log.Info("Create PersistentVolumeClaim", "name", pvc.Name)
	}

	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if err != nil {
		return err
	}

	if !metav1.IsControlledBy(pvc, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, pvc.Name)
		r.Recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return fmt.Errorf(msg)
	}

	// Step 2: Create Statefulset
	statefulset := &appsv1.StatefulSet{}
	err = r.Get(ctx, types.NamespacedName{Name: databaseName, Namespace: submarine.Namespace}, statefulset)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		statefulset = r.newSubmarineDatabaseStatefulSet(ctx, submarine)
		err = r.Create(ctx, statefulset)
		r.Log.Info("Create StatefulSet", "name", statefulset.Name)
	}

	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if err != nil {
		return err
	}

	if !metav1.IsControlledBy(statefulset, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, statefulset.Name)
		r.Recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return fmt.Errorf(msg)
	}

	if err != nil {
		return err
	}

	// Step 3: Create Service
	service := &corev1.Service{}
	err = r.Get(ctx, types.NamespacedName{Name: databaseName, Namespace: submarine.Namespace}, service)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		service = r.newSubmarineDatabaseService(ctx, submarine)
		err = r.Create(ctx, service)
		r.Log.Info("Create Service", "name", service.Name)
	}

	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if err != nil {
		return err
	}

	if !metav1.IsControlledBy(service, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, service.Name)
		r.Recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return fmt.Errorf(msg)
	}

	return nil
}
