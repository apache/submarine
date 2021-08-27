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
	traefikv1alpha1 "github.com/traefik/traefik/v2/pkg/provider/kubernetes/crd/traefik/v1alpha1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/klog/v2"
)

func newSubmarineMlflowPersistentVolumeClaim(submarine *v1alpha1.Submarine) *corev1.PersistentVolumeClaim {
	storageClassName := storageClassName
	return &corev1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: mlflowPvcName,
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
					corev1.ResourceStorage: resource.MustParse(submarine.Spec.Mlflow.StorageSize),
				},
			},
			StorageClassName: &storageClassName,
		},
	}
}

func newSubmarineMlflowDeployment(submarine *v1alpha1.Submarine) *appsv1.Deployment {
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: mlflowName,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(submarine, v1alpha1.SchemeGroupVersion.WithKind("Submarine")),
			},
		},
		Spec: appsv1.DeploymentSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"app": mlflowName,
				},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"app": mlflowName,
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:            mlflowName + "-container",
							Image:           "apache/submarine:mlflow-0.6.0-SNAPSHOT",
							ImagePullPolicy: "IfNotPresent",
							Ports: []corev1.ContainerPort{
								{
									ContainerPort: 5000,
								},
							},
							VolumeMounts: []corev1.VolumeMount{
								{
									MountPath: "/logs",
									Name:      "volume",
									SubPath:   mlflowName,
								},
							},
							ReadinessProbe: &corev1.Probe{
								Handler: corev1.Handler{
									TCPSocket: &corev1.TCPSocketAction{
										Port: intstr.FromInt(5000),
									},
								},
								InitialDelaySeconds: 60,
								PeriodSeconds:       10,
							},
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: "volume",
							VolumeSource: corev1.VolumeSource{
								PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
									ClaimName: mlflowPvcName,
								},
							},
						},
					},
				},
			},
		},
	}
}

func newSubmarineMlflowService(submarine *v1alpha1.Submarine) *corev1.Service {
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: mlflowServiceName,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(submarine, v1alpha1.SchemeGroupVersion.WithKind("Submarine")),
			},
		},
		Spec: corev1.ServiceSpec{
			Type: corev1.ServiceTypeClusterIP,
			Selector: map[string]string{
				"app": mlflowName,
			},
			Ports: []corev1.ServicePort{
				{
					Protocol:   "TCP",
					Port:       5000,
					TargetPort: intstr.FromInt(5000),
				},
			},
		},
	}
}

func newSubmarineMlflowIngressRoute(submarine *v1alpha1.Submarine) *traefikv1alpha1.IngressRoute {
	return &traefikv1alpha1.IngressRoute{
		ObjectMeta: metav1.ObjectMeta{
			Name: mlflowName + "-ingressroute",
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
					Match: "PathPrefix(`/mlflow`)",
					Services: []traefikv1alpha1.Service{
						{
							LoadBalancerSpec: traefikv1alpha1.LoadBalancerSpec{
								Kind: "Service",
								Name: mlflowServiceName,
								Port: 5000,
							},
						},
					},
				},
			},
		},
	}
}

// createSubmarineMlflow is a function to create submarine-mlflow.
// Reference: https://github.com/apache/submarine/blob/master/helm-charts/submarine/templates/submarine-mlflow.yaml
func (c *Controller) createSubmarineMlflow(submarine *v1alpha1.Submarine) error {
	klog.Info("[createSubmarineMlflow]")

	// Step 1: Create PersistentVolumeClaim
	pvc, err := c.persistentvolumeclaimLister.PersistentVolumeClaims(submarine.Namespace).Get(mlflowPvcName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		pvc, err = c.kubeclientset.CoreV1().PersistentVolumeClaims(submarine.Namespace).Create(context.TODO(),
			newSubmarineMlflowPersistentVolumeClaim(submarine),
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
	deployment, err := c.deploymentLister.Deployments(submarine.Namespace).Get(mlflowName)
	if errors.IsNotFound(err) {
		deployment, err = c.kubeclientset.AppsV1().Deployments(submarine.Namespace).Create(context.TODO(), newSubmarineMlflowDeployment(submarine), metav1.CreateOptions{})
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
	service, err := c.serviceLister.Services(submarine.Namespace).Get(mlflowServiceName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		service, err = c.kubeclientset.CoreV1().Services(submarine.Namespace).Create(context.TODO(), newSubmarineMlflowService(submarine), metav1.CreateOptions{})
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
	ingressroute, err := c.ingressrouteLister.IngressRoutes(submarine.Namespace).Get(mlflowIngressRouteName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		ingressroute, err = c.traefikclientset.TraefikV1alpha1().IngressRoutes(submarine.Namespace).Create(context.TODO(), newSubmarineMlflowIngressRoute(submarine), metav1.CreateOptions{})
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
