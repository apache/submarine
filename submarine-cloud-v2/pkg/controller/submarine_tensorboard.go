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

	traefikv1alpha1 "github.com/traefik/traefik/v2/pkg/provider/kubernetes/crd/traefik/v1alpha1"
)

func newSubmarineTensorboardPersistentVolume(submarine *v1alpha1.Submarine, pvName string, storageSize string, persistentVolumeSource *corev1.PersistentVolumeSource) *corev1.PersistentVolume {
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
				corev1.ResourceStorage: resource.MustParse(storageSize),
			},
			PersistentVolumeSource: *persistentVolumeSource,
		},
	}
}

func newSubmarineTensorboardPersistentVolumeClaim(submarine *v1alpha1.Submarine, pvcName string, pvName string, storageSize string) *corev1.PersistentVolumeClaim {
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
					corev1.ResourceStorage: resource.MustParse(storageSize),
				},
			},
			VolumeName:       pvName,
			StorageClassName: &storageClassName,
		},
	}
}

func newSubmarineTensorboardDeployment(submarine *v1alpha1.Submarine, tensorboardName string, pvcName string) *appsv1.Deployment {
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: tensorboardName,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(submarine, v1alpha1.SchemeGroupVersion.WithKind("Submarine")),
			},
		},
		Spec: appsv1.DeploymentSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"app": tensorboardName + "-pod",
				},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"app": tensorboardName + "-pod",
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  tensorboardName + "-container",
							Image: "tensorflow/tensorflow:1.11.0",
							Command: []string{
								"tensorboard",
								"--logdir=/logs",
								"--path_prefix=/tensorboard",
							},
							ImagePullPolicy: "IfNotPresent",
							Ports: []corev1.ContainerPort{
								{
									ContainerPort: 6006,
								},
							},
							VolumeMounts: []corev1.VolumeMount{
								{
									MountPath: "/logs",
									Name:      "volume",
									SubPath:   tensorboardName,
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

func newSubmarineTensorboardService(submarine *v1alpha1.Submarine, serviceName string, tensorboardName string) *corev1.Service {
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: serviceName,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(submarine, v1alpha1.SchemeGroupVersion.WithKind("Submarine")),
			},
		},
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{
				"app": tensorboardName + "-pod",
			},
			Ports: []corev1.ServicePort{
				{
					Protocol:   "TCP",
					Port:       8080,
					TargetPort: intstr.FromInt(6006),
				},
			},
		},
	}
}

func newSubmarineTensorboardIngressRoute(submarine *v1alpha1.Submarine, tensorboardName string, serviceName string) *traefikv1alpha1.IngressRoute {
	return &traefikv1alpha1.IngressRoute{
		ObjectMeta: metav1.ObjectMeta{
			Name: tensorboardName + "-ingressroute",
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(submarine, v1alpha1.SchemeGroupVersion.WithKind("Submarine")),
			},
		},
		Spec: traefikv1alpha1.IngressRouteSpec{
			EntryPoints: []string{
				"web",
			},
			Routes: []traefikv1alpha1.Route{
				{
					Kind:  "Rule",
					Match: "PathPrefix(`/tensorboard`)",
					Services: []traefikv1alpha1.Service{
						{
							LoadBalancerSpec: traefikv1alpha1.LoadBalancerSpec{
								Kind: "Service",
								Name: serviceName,
								Port: 8080,
							},
						},
					},
				},
			},
		},
	}
}

// createSubmarineTensorboard is a function to create submarine-tensorboard.
// Reference: https://github.com/apache/submarine/blob/master/helm-charts/submarine/templates/submarine-tensorboard.yaml
func (c *Controller) createSubmarineTensorboard(submarine *v1alpha1.Submarine, namespace string, spec *v1alpha1.SubmarineSpec) error {
	klog.Info("[createSubmarineTensorboard]")
	tensorboardName := "submarine-tensorboard"

	// Step 1: Create PersistentVolume
	// PersistentVolumes are not namespaced resources, so we add the namespace
	// as a suffix to distinguish them
	pvName := tensorboardName + "-pv--" + namespace
	pv, pv_err := c.persistentvolumeLister.Get(pvName)

	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(pv_err) {
		var persistentVolumeSource corev1.PersistentVolumeSource
		switch spec.Storage.StorageType {
		case "nfs":
			persistentVolumeSource = corev1.PersistentVolumeSource{
				NFS: &corev1.NFSVolumeSource{
					Server: spec.Storage.NfsIP,
					Path:   spec.Storage.NfsPath,
				},
			}
		case "host":
			hostPathType := corev1.HostPathDirectoryOrCreate
			persistentVolumeSource = corev1.PersistentVolumeSource{
				HostPath: &corev1.HostPathVolumeSource{
					Path: spec.Storage.HostPath,
					Type: &hostPathType,
				},
			}
		default:
			klog.Warningln("	Invalid storageType found in submarine spec, nothing will be created!")
			return nil
		}
		pv, pv_err = c.kubeclientset.CoreV1().PersistentVolumes().Create(context.TODO(), newSubmarineTensorboardPersistentVolume(submarine, pvName, spec.Tensorboard.StorageSize, &persistentVolumeSource), metav1.CreateOptions{})
		if pv_err != nil {
			klog.Info(pv_err)
		}
		klog.Info("	Create PersistentVolume: ", pv.Name)
	}

	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if pv_err != nil {
		return pv_err
	}

	if !metav1.IsControlledBy(pv, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, pv.Name)
		c.recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return fmt.Errorf(msg)
	}

	// Step 2: Create PersistentVolumeClaim
	pvcName := tensorboardName + "-pvc"
	pvc, pvc_err := c.persistentvolumeclaimLister.PersistentVolumeClaims(namespace).Get(pvcName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(pvc_err) {
		pvc, pvc_err = c.kubeclientset.CoreV1().PersistentVolumeClaims(namespace).Create(context.TODO(),
			newSubmarineTensorboardPersistentVolumeClaim(submarine, pvcName, pvName, spec.Tensorboard.StorageSize),
			metav1.CreateOptions{})
		if pvc_err != nil {
			klog.Info(pvc_err)
		}
		klog.Info("	Create PersistentVolumeClaim: ", pvc.Name)
	}
	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if pvc_err != nil {
		return pvc_err
	}

	if !metav1.IsControlledBy(pvc, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, pvc.Name)
		c.recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return fmt.Errorf(msg)
	}

	// Step 3: Create Deployment
	deployment, deployment_err := c.deploymentLister.Deployments(namespace).Get(tensorboardName)
	if errors.IsNotFound(deployment_err) {
		deployment, deployment_err = c.kubeclientset.AppsV1().Deployments(namespace).Create(context.TODO(), newSubmarineTensorboardDeployment(submarine, tensorboardName, pvcName), metav1.CreateOptions{})
		if deployment_err != nil {
			klog.Info(deployment_err)
		}
		klog.Info("	Create Deployment: ", deployment.Name)
	}
	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if deployment_err != nil {
		return deployment_err
	}

	if !metav1.IsControlledBy(deployment, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, deployment.Name)
		c.recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return fmt.Errorf(msg)
	}

	// Step 4: Create Service
	serviceName := tensorboardName + "-service"
	service, service_err := c.serviceLister.Services(namespace).Get(serviceName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(service_err) {
		service, service_err = c.kubeclientset.CoreV1().Services(namespace).Create(context.TODO(), newSubmarineTensorboardService(submarine, serviceName, tensorboardName), metav1.CreateOptions{})
		if service_err != nil {
			klog.Info(service_err)
		}
		klog.Info(" Create Service: ", service.Name)
	}
	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if service_err != nil {
		return service_err
	}

	if !metav1.IsControlledBy(service, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, service.Name)
		c.recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return fmt.Errorf(msg)
	}

	// Step 5: Create IngressRoute
	ingressroute, ingressroute_err := c.ingressrouteLister.IngressRoutes(namespace).Get(tensorboardName + "-ingressroute")
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(ingressroute_err) {
		ingressroute, ingressroute_err = c.traefikclientset.TraefikV1alpha1().IngressRoutes(namespace).Create(context.TODO(), newSubmarineTensorboardIngressRoute(submarine, tensorboardName, serviceName), metav1.CreateOptions{})
		if ingressroute_err != nil {
			klog.Info(ingressroute_err)
		}
		klog.Info(" Create IngressRoute: ", ingressroute.Name)
	}
	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if ingressroute_err != nil {
		return ingressroute_err
	}

	if !metav1.IsControlledBy(ingressroute, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, ingressroute.Name)
		c.recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return fmt.Errorf(msg)
	}

	return nil
}
