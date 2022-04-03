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

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
)

func newSubmarineDatabasePersistentVolumeClaim(submarine *v1alpha1.Submarine) *corev1.PersistentVolumeClaim {
	pvc, err := ParsePersistentVolumeClaimYaml(databaseYamlPath)
	if err != nil {
		klog.Info("[Error] ParsePersistentVolumeClaim", err)
	}
	pvc.ObjectMeta.OwnerReferences = []metav1.OwnerReference{
		*metav1.NewControllerRef(submarine, v1alpha1.SchemeGroupVersion.WithKind("Submarine")),
	}

	return pvc
}

func newSubmarineDatabaseDeployment(submarine *v1alpha1.Submarine) *appsv1.Deployment {
	databaseImage := submarine.Spec.Database.Image

	deployment, err := ParseDeploymentYaml(databaseYamlPath)
	if err != nil {
		klog.Info("[Error] ParseDeploymentYaml", err)
	}
	deployment.ObjectMeta.OwnerReferences = []metav1.OwnerReference{
		*metav1.NewControllerRef(submarine, v1alpha1.SchemeGroupVersion.WithKind("Submarine")),
	}
	if databaseImage != "" {
		deployment.Spec.Template.Spec.Containers[0].Image = databaseImage
	}

	return deployment
}

func newSubmarineDatabaseStatefulSet(submarine *v1alpha1.Submarine) *appsv1.StatefulSet {
	statefulset, err := ParseStatefulSetYaml(databaseYamlPath)
	if err != nil {
		klog.Info("[Error] ParseStatefulSetYaml", err)
	}
	statefulset.ObjectMeta.OwnerReferences = []metav1.OwnerReference{
		*metav1.NewControllerRef(submarine, v1alpha1.SchemeGroupVersion.WithKind("Submarine")),
	}

	databaseImage := submarine.Spec.Database.Image
	if databaseImage != "" {
		statefulset.Spec.Template.Spec.Containers[0].Image = databaseImage
	}

	return statefulset
}

func newSubmarineDatabaseService(submarine *v1alpha1.Submarine) *corev1.Service {
	service, err := ParseServiceYaml(databaseYamlPath)
	if err != nil {
		klog.Info("[Error] ParseServiceYaml", err)
	}
	service.ObjectMeta.OwnerReferences = []metav1.OwnerReference{
		*metav1.NewControllerRef(submarine, v1alpha1.SchemeGroupVersion.WithKind("Submarine")),
	}
	return service
}

// createSubmarineDatabase is a function to create submarine-database.
// Reference: https://github.com/apache/submarine/blob/master/helm-charts/submarine/templates/submarine-database.yaml
func (c *Controller) createSubmarineDatabase(submarine *v1alpha1.Submarine) error {
	klog.Info("[createSubmarineDatabase]")

	// Step 1: Create PersistentVolumeClaim
	pvc, err := c.persistentvolumeclaimLister.PersistentVolumeClaims(submarine.Namespace).Get(databasePvcName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		pvc, err = c.kubeclientset.CoreV1().PersistentVolumeClaims(submarine.Namespace).Create(context.TODO(), newSubmarineDatabasePersistentVolumeClaim(submarine), metav1.CreateOptions{})
		if err != nil {
			klog.Info(err)
		}
		klog.Info("	Create PersistentVolumeClaim: ", pvc.Name)
	}
	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if err != nil {
		return err
	}

	if !metav1.IsControlledBy(pvc, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, pvc.Name)
		c.recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return fmt.Errorf(msg)
	}

	// Step 2: Create Deployment
	statefulset, err := c.statefulsetLister.StatefulSets(submarine.Namespace).Get(databaseName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		statefulset, err = c.kubeclientset.AppsV1().StatefulSets(submarine.Namespace).Create(context.TODO(), newSubmarineDatabaseStatefulSet(submarine), metav1.CreateOptions{})
		if err != nil {
			klog.Info(err)
		}
		klog.Info("	Create StatefulSet: ", statefulset.Name)
	}
	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if err != nil {
		return err
	}

	if !metav1.IsControlledBy(statefulset, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, statefulset.Name)
		c.recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return fmt.Errorf(msg)
	}

	if err != nil {
		return err
	}

	// Step 3: Create Service
	service, err := c.serviceLister.Services(submarine.Namespace).Get(databaseName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		service, err = c.kubeclientset.CoreV1().Services(submarine.Namespace).Create(context.TODO(), newSubmarineDatabaseService(submarine), metav1.CreateOptions{})
		if err != nil {
			klog.Info(err)
		}
		klog.Info("	Create Service: ", service.Name)
	}
	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if err != nil {
		return err
	}

	if !metav1.IsControlledBy(service, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, service.Name)
		c.recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return fmt.Errorf(msg)
	}

	return nil
}
