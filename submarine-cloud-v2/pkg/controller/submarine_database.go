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
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/klog/v2"
)

func newSubmarineDatabasePersistentVolumeClaim(submarine *v1alpha1.Submarine) *corev1.PersistentVolumeClaim {
	storageClassName := storageClassName
	return &corev1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: databasePvcName,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(submarine, v1alpha1.SchemeGroupVersion.WithKind("Submarine")),
			},
		},
		Spec: corev1.PersistentVolumeClaimSpec{
			AccessModes: []corev1.PersistentVolumeAccessMode{
				corev1.ReadWriteOnce,
			},
			Resources: corev1.ResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceStorage: resource.MustParse(submarine.Spec.Database.StorageSize),
				},
			},
			StorageClassName: &storageClassName,
		},
	}
}

func newSubmarineDatabaseDeployment(submarine *v1alpha1.Submarine) *appsv1.Deployment {
	databaseImage := submarine.Spec.Database.Image
	if databaseImage == "" {
		databaseImage = "apache/submarine:database-" + submarine.Spec.Version
	}

	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: databaseName,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(submarine, v1alpha1.SchemeGroupVersion.WithKind("Submarine")),
			},
		},
		Spec: appsv1.DeploymentSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"app": databaseName,
				},
			},
			Replicas: submarine.Spec.Database.Replicas,
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"app": databaseName,
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:            databaseName,
							Image:           databaseImage,
							ImagePullPolicy: "IfNotPresent",
							Ports: []corev1.ContainerPort{
								{
									ContainerPort: 3306,
								},
							},
							Env: []corev1.EnvVar{
								{
									Name:  "MYSQL_ROOT_PASSWORD",
									Value: "password",
								},
							},
							VolumeMounts: []corev1.VolumeMount{
								{
									MountPath: "/var/lib/mysql",
									Name:      "volume",
									SubPath:   databaseName,
								},
							},
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: "volume",
							VolumeSource: corev1.VolumeSource{
								PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
									ClaimName: databasePvcName,
								},
							},
						},
					},
				},
			},
		},
	}
}

func newSubmarineDatabaseService(submarine *v1alpha1.Submarine) *corev1.Service {
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: databaseName,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(submarine, v1alpha1.SchemeGroupVersion.WithKind("Submarine")),
			},
		},
		Spec: corev1.ServiceSpec{
			Ports: []corev1.ServicePort{
				{
					Port:       3306,
					TargetPort: intstr.FromInt(3306),
					Name:       databaseName,
				},
			},
			Selector: map[string]string{
				"app": databaseName,
			},
		},
	}
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
	deployment, err := c.deploymentLister.Deployments(submarine.Namespace).Get(databaseName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		deployment, err = c.kubeclientset.AppsV1().Deployments(submarine.Namespace).Create(context.TODO(), newSubmarineDatabaseDeployment(submarine), metav1.CreateOptions{})
		if err != nil {
			klog.Info(err)
		}
		klog.Info("	Create Deployment: ", deployment.Name)
	}
	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if err != nil {
		return err
	}

	if !metav1.IsControlledBy(deployment, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, deployment.Name)
		c.recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return fmt.Errorf(msg)
	}

	// Update the replicas of the database deployment if it is not equal to spec
	if submarine.Spec.Database.Replicas != nil && *submarine.Spec.Database.Replicas != *deployment.Spec.Replicas {
		klog.V(4).Infof("Submarine %s database spec replicas: %d, actual replicas: %d", submarine.Name, *submarine.Spec.Database.Replicas, *deployment.Spec.Replicas)
		_, err = c.kubeclientset.AppsV1().Deployments(submarine.Namespace).Update(context.TODO(), newSubmarineDatabaseDeployment(submarine), metav1.UpdateOptions{})
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
