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

func newSubmarineDatabasePersistentVolume(submarine *v1alpha1.Submarine, persistentVolumeSource *corev1.PersistentVolumeSource, pvName string) *corev1.PersistentVolume {
	return &corev1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: pvName,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(submarine, v1alpha1.SchemeGroupVersion.WithKind("Submarine")),
			},
		},
		Spec: corev1.PersistentVolumeSpec{
			AccessModes: []corev1.PersistentVolumeAccessMode{
				corev1.ReadWriteMany,
			},
			Capacity: corev1.ResourceList{
				corev1.ResourceStorage: resource.MustParse(submarine.Spec.Database.StorageSize),
			},
			PersistentVolumeSource: *persistentVolumeSource,
		},
	}
}

func newSubmarineDatabasePersistentVolumeClaim(submarine *v1alpha1.Submarine, pvcName string, pvName string) *corev1.PersistentVolumeClaim {
	storageClassName := ""
	return &corev1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: pvcName,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(submarine, v1alpha1.SchemeGroupVersion.WithKind("Submarine")),
			},
		},
		Spec: corev1.PersistentVolumeClaimSpec{
			AccessModes: []corev1.PersistentVolumeAccessMode{
				corev1.ReadWriteMany,
			},
			Resources: corev1.ResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceStorage: resource.MustParse(submarine.Spec.Database.StorageSize),
				},
			},
			VolumeName:       pvName,
			StorageClassName: &storageClassName,
		},
	}
}

func newSubmarineDatabaseDeployment(submarine *v1alpha1.Submarine, pvcName string) *appsv1.Deployment {
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
									ClaimName: pvcName,
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
func (c *Controller) createSubmarineDatabase(submarine *v1alpha1.Submarine, namespace string) (*appsv1.Deployment, error) {
	klog.Info("[createSubmarineDatabase]")

	// Step1: Create PersistentVolume
	// PersistentVolumes are not namespaced resources, so we add the namespace
	// as a suffix to distinguish them
	pvName := databaseName + "-pv--" + namespace
	pv, pv_err := c.persistentvolumeLister.Get(pvName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(pv_err) {
		var persistentVolumeSource corev1.PersistentVolumeSource
		switch submarine.Spec.Storage.StorageType {
		case "nfs":
			persistentVolumeSource = corev1.PersistentVolumeSource{
				NFS: &corev1.NFSVolumeSource{
					Server: submarine.Spec.Storage.NfsIP,
					Path:   submarine.Spec.Storage.NfsPath,
				},
			}
		case "host":
			hostPathType := corev1.HostPathDirectoryOrCreate
			persistentVolumeSource = corev1.PersistentVolumeSource{
				HostPath: &corev1.HostPathVolumeSource{
					Path: submarine.Spec.Storage.HostPath,
					Type: &hostPathType,
				},
			}
		default:
			klog.Warningln("	Invalid storageType found in submarine spec, nothing will be created!")
			return nil, nil
		}
		pv, pv_err = c.kubeclientset.CoreV1().PersistentVolumes().Create(context.TODO(), newSubmarineDatabasePersistentVolume(submarine, &persistentVolumeSource, pvName), metav1.CreateOptions{})
		if pv_err != nil {
			klog.Info(pv_err)
		}
		klog.Info("	Create PersistentVolume: ", pv.Name)
	}
	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if pv_err != nil {
		return nil, pv_err
	}

	if !metav1.IsControlledBy(pv, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, pv.Name)
		c.recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return nil, fmt.Errorf(msg)
	}

	// Step2: Create PersistentVolumeClaim
	pvcName := databaseName + "-pvc"
	pvc, pvc_err := c.persistentvolumeclaimLister.PersistentVolumeClaims(namespace).Get(pvcName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(pvc_err) {
		pvc, pvc_err = c.kubeclientset.CoreV1().PersistentVolumeClaims(namespace).Create(context.TODO(), newSubmarineDatabasePersistentVolumeClaim(submarine, pvcName, pvName), metav1.CreateOptions{})
		if pvc_err != nil {
			klog.Info(pvc_err)
		}
		klog.Info("	Create PersistentVolumeClaim: ", pvc.Name)
	}
	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if pvc_err != nil {
		return nil, pvc_err
	}

	if !metav1.IsControlledBy(pvc, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, pvc.Name)
		c.recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return nil, fmt.Errorf(msg)
	}

	// Step3: Create Deployment
	deployment, deployment_err := c.deploymentLister.Deployments(namespace).Get(databaseName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(deployment_err) {
		deployment, deployment_err = c.kubeclientset.AppsV1().Deployments(namespace).Create(context.TODO(), newSubmarineDatabaseDeployment(submarine, pvcName), metav1.CreateOptions{})
		if deployment_err != nil {
			klog.Info(deployment_err)
		}
		klog.Info("	Create Deployment: ", deployment.Name)
	}
	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if deployment_err != nil {
		return nil, deployment_err
	}

	if !metav1.IsControlledBy(deployment, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, deployment.Name)
		c.recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return nil, fmt.Errorf(msg)
	}

	// Update the replicas of the database deployment if it is not equal to spec
	if submarine.Spec.Database.Replicas != nil && *submarine.Spec.Database.Replicas != *deployment.Spec.Replicas {
		klog.V(4).Infof("Submarine %s database spec replicas: %d, actual replicas: %d", submarine.Name, *submarine.Spec.Database.Replicas, *deployment.Spec.Replicas)
		deployment, deployment_err = c.kubeclientset.AppsV1().Deployments(submarine.Namespace).Update(context.TODO(), newSubmarineDatabaseDeployment(submarine, pvcName), metav1.UpdateOptions{})
	}

	if deployment_err != nil {
		return nil, deployment_err
	}

	// Step4: Create Service
	service, service_err := c.serviceLister.Services(namespace).Get(databaseName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(service_err) {
		service, service_err = c.kubeclientset.CoreV1().Services(namespace).Create(context.TODO(), newSubmarineDatabaseService(submarine), metav1.CreateOptions{})
		if service_err != nil {
			klog.Info(service_err)
		}
		klog.Info("	Create Service: ", service.Name)
	}
	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if service_err != nil {
		return nil, service_err
	}

	if !metav1.IsControlledBy(service, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, service.Name)
		c.recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return nil, fmt.Errorf(msg)
	}

	return deployment, nil
}
