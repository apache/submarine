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

func newSubmarineTensorboardPersistentVolumeClaim(submarine *v1alpha1.Submarine) *corev1.PersistentVolumeClaim {
	storageClassName := tensorboardScName
	return &corev1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: tensorboardPvcName,
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
					corev1.ResourceStorage: resource.MustParse(submarine.Spec.Tensorboard.StorageSize),
				},
			},
			StorageClassName: &storageClassName,
		},
	}
}

func newSubmarineTensorboardDeployment(submarine *v1alpha1.Submarine) *appsv1.Deployment {
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
							ReadinessProbe: &corev1.Probe{
								Handler: corev1.Handler{
									TCPSocket: &corev1.TCPSocketAction{
										Port: intstr.FromInt(6006),
									},
								},
								PeriodSeconds: 10,
							},
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: "volume",
							VolumeSource: corev1.VolumeSource{
								PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
									ClaimName: tensorboardPvcName,
								},
							},
						},
					},
				},
			},
		},
	}
}

func newSubmarineTensorboardService(submarine *v1alpha1.Submarine) *corev1.Service {
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: tensorboardServiceName,
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

func newSubmarineTensorboardIngressRoute(submarine *v1alpha1.Submarine) *traefikv1alpha1.IngressRoute {
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
								Name: tensorboardServiceName,
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
func (c *Controller) createSubmarineTensorboard(submarine *v1alpha1.Submarine) error {
	klog.Info("[createSubmarineTensorboard]")

	// Step 1: Create PersistentVolumeClaim
	pvc, err := c.persistentvolumeclaimLister.PersistentVolumeClaims(submarine.Namespace).Get(tensorboardPvcName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		pvc, err = c.kubeclientset.CoreV1().PersistentVolumeClaims(submarine.Namespace).Create(context.TODO(),
			newSubmarineTensorboardPersistentVolumeClaim(submarine),
			metav1.CreateOptions{})
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
	deployment, err := c.deploymentLister.Deployments(submarine.Namespace).Get(tensorboardName)
	if errors.IsNotFound(err) {
		deployment, err = c.kubeclientset.AppsV1().Deployments(submarine.Namespace).Create(context.TODO(), newSubmarineTensorboardDeployment(submarine), metav1.CreateOptions{})
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

	// Step 3: Create Service
	service, err := c.serviceLister.Services(submarine.Namespace).Get(tensorboardServiceName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		service, err = c.kubeclientset.CoreV1().Services(submarine.Namespace).Create(context.TODO(), newSubmarineTensorboardService(submarine), metav1.CreateOptions{})
		if err != nil {
			klog.Info(err)
		}
		klog.Info(" Create Service: ", service.Name)
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

	// Step 4: Create IngressRoute
	ingressroute, err := c.ingressrouteLister.IngressRoutes(submarine.Namespace).Get(tensorboardIngressRouteName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		ingressroute, err = c.traefikclientset.TraefikV1alpha1().IngressRoutes(submarine.Namespace).Create(context.TODO(), newSubmarineTensorboardIngressRoute(submarine), metav1.CreateOptions{})
		if err != nil {
			klog.Info(err)
		}
		klog.Info(" Create IngressRoute: ", ingressroute.Name)
	}
	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if err != nil {
		return err
	}

	if !metav1.IsControlledBy(ingressroute, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, ingressroute.Name)
		c.recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return fmt.Errorf(msg)
	}

	return nil
}
