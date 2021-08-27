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
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/klog/v2"
)

func newSubmarineServerServiceAccount(submarine *v1alpha1.Submarine) *corev1.ServiceAccount {
	return &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name: serverName,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(submarine, v1alpha1.SchemeGroupVersion.WithKind("Submarine")),
			},
		},
	}
}

func newSubmarineServerService(submarine *v1alpha1.Submarine) *corev1.Service {
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: serverName,
			Labels: map[string]string{
				"app": serverName,
			},
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(submarine, v1alpha1.SchemeGroupVersion.WithKind("Submarine")),
			},
		},
		Spec: corev1.ServiceSpec{
			Ports: []corev1.ServicePort{
				{
					Port:       8080,
					TargetPort: intstr.FromInt(8080),
					Protocol:   "TCP",
				},
			},
			Selector: map[string]string{
				"app": serverName,
			},
		},
	}
}

func newSubmarineServerDeployment(submarine *v1alpha1.Submarine) *appsv1.Deployment {
	serverImage := submarine.Spec.Server.Image
	serverReplicas := *submarine.Spec.Server.Replicas
	if serverImage == "" {
		serverImage = "apache/submarine:server-" + submarine.Spec.Version
	}

	ownerReference := *metav1.NewControllerRef(submarine, v1alpha1.SchemeGroupVersion.WithKind("Submarine"))

	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: serverName,
			OwnerReferences: []metav1.OwnerReference{
				ownerReference,
			},
		},
		Spec: appsv1.DeploymentSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"app": serverName,
				},
			},
			Replicas: &serverReplicas,
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"app": serverName,
					},
				},
				Spec: corev1.PodSpec{
					ServiceAccountName: serverName,
					Containers: []corev1.Container{
						{
							Name:  serverName,
							Image: serverImage,
							Env: []corev1.EnvVar{
								{
									Name:  "SUBMARINE_SERVER_PORT",
									Value: "8080",
								},
								{
									Name:  "SUBMARINE_SERVER_PORT_8080_TCP",
									Value: "8080",
								},
								{
									Name:  "SUBMARINE_SERVER_DNS_NAME",
									Value: serverName + "." + submarine.Namespace,
								},
								{
									Name:  "K8S_APISERVER_URL",
									Value: "kubernetes.default.svc",
								},
								{
									Name:  "ENV_NAMESPACE",
									Value: submarine.Namespace,
								},
								{
									Name:  "SUBMARINE_APIVERSION",
									Value: ownerReference.APIVersion,
								},
								{
									Name:  "SUBMARINE_KIND",
									Value: ownerReference.Kind,
								},
								{
									Name:  "SUBMARINE_NAME",
									Value: ownerReference.Name,
								},
								{
									Name:  "SUBMARINE_UID",
									Value: string(ownerReference.UID),
								},
							},
							Ports: []corev1.ContainerPort{
								{
									ContainerPort: 8080,
								},
							},
							ImagePullPolicy: "IfNotPresent",
						},
					},
				},
			},
		},
	}
}

// createSubmarineServer is a function to create submarine-server.
// Reference: https://github.com/apache/submarine/blob/master/helm-charts/submarine/templates/submarine-server.yaml
func (c *Controller) createSubmarineServer(submarine *v1alpha1.Submarine) error {
	klog.Info("[createSubmarineServer]")

	// Step1: Create ServiceAccount
	serviceaccount, err := c.serviceaccountLister.ServiceAccounts(submarine.Namespace).Get(serverName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		serviceaccount, err = c.kubeclientset.CoreV1().ServiceAccounts(submarine.Namespace).Create(context.TODO(), newSubmarineServerServiceAccount(submarine), metav1.CreateOptions{})
		klog.Info("	Create ServiceAccount: ", serviceaccount.Name)
	}

	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if err != nil {
		return err
	}

	if !metav1.IsControlledBy(serviceaccount, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, serviceaccount.Name)
		c.recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return fmt.Errorf(msg)
	}

	// Step2: Create Service
	service, err := c.serviceLister.Services(submarine.Namespace).Get(serverName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		service, err = c.kubeclientset.CoreV1().Services(submarine.Namespace).Create(context.TODO(), newSubmarineServerService(submarine), metav1.CreateOptions{})
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

	// Step3: Create Deployment
	deployment, err := c.deploymentLister.Deployments(submarine.Namespace).Get(serverName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		deployment, err = c.kubeclientset.AppsV1().Deployments(submarine.Namespace).Create(context.TODO(), newSubmarineServerDeployment(submarine), metav1.CreateOptions{})
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

	// Update the replicas of the server deployment if it is not equal to spec
	if submarine.Spec.Server.Replicas != nil && *submarine.Spec.Server.Replicas != *deployment.Spec.Replicas {
		klog.V(4).Infof("Submarine %s server spec replicas: %d, actual replicas: %d", submarine.Name, *submarine.Spec.Server.Replicas, *deployment.Spec.Replicas)
		_, err = c.kubeclientset.AppsV1().Deployments(submarine.Namespace).Update(context.TODO(), newSubmarineServerDeployment(submarine), metav1.UpdateOptions{})
	}

	if err != nil {
		return err
	}

	return nil
}
