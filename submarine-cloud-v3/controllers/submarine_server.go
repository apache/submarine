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

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	submarineapacheorgv1alpha1 "github.com/apache/submarine/submarine-cloud-v3/api/v1alpha1"

	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

func (r *SubmarineReconciler) newSubmarineServerServiceAccount(ctx context.Context, submarine *submarineapacheorgv1alpha1.Submarine) *corev1.ServiceAccount {
	serviceAccount, err := ParseServiceAccountYaml(serverYamlPath)
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

func (r *SubmarineReconciler) newSubmarineServerService(ctx context.Context, submarine *submarineapacheorgv1alpha1.Submarine) *corev1.Service {
	service, err := ParseServiceYaml(serverYamlPath)
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

func (r *SubmarineReconciler) newSubmarineServerDeployment(ctx context.Context, submarine *submarineapacheorgv1alpha1.Submarine) *appsv1.Deployment {
	serverReplicas := *submarine.Spec.Server.Replicas
	operatorEnv := []corev1.EnvVar{
		{
			Name:  "SUBMARINE_SERVER_DNS_NAME",
			Value: serverName + "." + submarine.Namespace,
		},
		{
			Name:  "ENV_NAMESPACE",
			Value: submarine.Namespace,
		},
		{
			Name:  "SUBMARINE_APIVERSION",
			Value: submarine.APIVersion,
		},
		{
			Name:  "SUBMARINE_KIND",
			Value: submarine.Kind,
		},
		{
			Name:  "SUBMARINE_NAME",
			Value: submarine.Name,
		},
		{
			Name:  "SUBMARINE_UID",
			Value: string(submarine.UID),
		},
	}
	// extra envs
	extraEnv := submarine.Spec.Server.Env
	if extraEnv != nil {
		operatorEnv = append(operatorEnv, extraEnv...)
	}

	deployment, err := ParseDeploymentYaml(serverYamlPath)
	if err != nil {
		r.Log.Error(err, "ParseDeploymentYaml")
	}
	deployment.Namespace = submarine.Namespace
	err = controllerutil.SetControllerReference(submarine, deployment, r.Scheme)
	if err != nil {
		r.Log.Error(err, "Set Deployment ControllerReference")
	}
	deployment.Spec.Replicas = &serverReplicas
	deployment.Spec.Template.Spec.Containers[0].Env = append(deployment.Spec.Template.Spec.Containers[0].Env, operatorEnv...)

	// server image
	serverImage := submarine.Spec.Server.Image
	if serverImage != "" {
		deployment.Spec.Template.Spec.Containers[0].Image = serverImage
	} else {
		deployment.Spec.Template.Spec.Containers[0].Image = fmt.Sprintf("apache/submarine:server-%s", submarine.Spec.Version)
	}
	// minio/mc image
	mcImage := submarine.Spec.Common.Image.McImage
	if mcImage != "" {
		deployment.Spec.Template.Spec.InitContainers[0].Image = mcImage
	}
	// pull secrets
	pullSecrets := submarine.Spec.Common.Image.PullSecrets
	if pullSecrets != nil {
		deployment.Spec.Template.Spec.ImagePullSecrets = r.CreatePullSecrets(&pullSecrets)
	}

	return deployment
}

// createSubmarineServer is a function to create submarine-server.
// Reference: https://github.com/apache/submarine/blob/master/submarine-cloud-v3/artifacts/submarine-server.yaml
func (r *SubmarineReconciler) createSubmarineServer(ctx context.Context, submarine *submarineapacheorgv1alpha1.Submarine) error {
	r.Log.Info("Enter createSubmarineServer")

	// Step1: Create ServiceAccount
	serviceaccount := &corev1.ServiceAccount{}
	err := r.Get(ctx, types.NamespacedName{Name: serverName, Namespace: submarine.Namespace}, serviceaccount)

	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		serviceaccount = r.newSubmarineServerServiceAccount(ctx, submarine)
		err = r.Create(ctx, serviceaccount)
		r.Log.Info("Create ServiceAccount", "name", serviceaccount.Name)
	}

	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if err != nil {
		return err
	}

	if !metav1.IsControlledBy(serviceaccount, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, serviceaccount.Name)
		r.Recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return fmt.Errorf(msg)
	}

	// Step2: Create Service
	service := &corev1.Service{}
	err = r.Get(ctx, types.NamespacedName{Name: serverName, Namespace: submarine.Namespace}, service)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		service = r.newSubmarineServerService(ctx, submarine)
		err = r.Create(ctx, service)
		r.Log.Info("Create Service", "name", service.Name)
	}

	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if err != nil {
		if errors.IsNotFound(err) {
			return nil
		}
		return err
	}

	if !metav1.IsControlledBy(service, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, service.Name)
		r.Recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return fmt.Errorf(msg)
	}

	// Step3: Create Deployment
	deployment := &appsv1.Deployment{}
	err = r.Get(ctx, types.NamespacedName{Name: serverName, Namespace: submarine.Namespace}, deployment)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		deployment = r.newSubmarineServerDeployment(ctx, submarine)
		err = r.Create(ctx, deployment)
		r.Log.Info("Create Deployment", "name", deployment.Name)
	}

	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if err != nil {
		if errors.IsNotFound(err) {
			return nil
		}
		return err
	}

	if !metav1.IsControlledBy(deployment, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, deployment.Name)
		r.Recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return fmt.Errorf(msg)
	}

	// Update the replicas of the server deployment if it is not equal to spec
	if submarine.Spec.Server.Replicas != nil && *submarine.Spec.Server.Replicas != *deployment.Spec.Replicas {
		msg := fmt.Sprintf("Submarine %s server spec replicas", submarine.Name)
		r.Log.Info(msg, "server spec", *submarine.Spec.Server.Replicas, "actual", *deployment.Spec.Replicas)

		deployment = r.newSubmarineServerDeployment(ctx, submarine)
		err = r.Update(ctx, deployment)
	}

	if err != nil {
		return err
	}

	return nil
}
