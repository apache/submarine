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
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	submarineapacheorgv1alpha1 "github.com/apache/submarine/submarine-cloud-v3/api/v1alpha1"
)

func (r *SubmarineReconciler) newSubmarineAgentDeployment(ctx context.Context, submarine *submarineapacheorgv1alpha1.Submarine) *appsv1.Deployment {
	deployment, err := util.ParseDeploymentYaml(agentYamlPath)
	if err != nil {
		r.Log.Error(err, "ParseDeploymentYaml")
	}
	deployment.Namespace = submarine.Namespace
	err = controllerutil.SetControllerReference(submarine, deployment, r.Scheme)
	if err != nil {
		r.Log.Error(err, "Set Deployment ControllerReference")
	}

	// env
	env := []corev1.EnvVar{
		{
			Name:  "SUBMARINE_UID",
			Value: string(submarine.UID),
		},
	}
	deployment.Spec.Template.Spec.Containers[0].Env = append(deployment.Spec.Template.Spec.Containers[0].Env, env...)

	// agent image
	if submarine.Spec.Agent != nil {
		agentImage := submarine.Spec.Agent.Image
		if agentImage != "" {
			deployment.Spec.Template.Spec.Containers[0].Image = agentImage
		} else {
			deployment.Spec.Template.Spec.Containers[0].Image = fmt.Sprintf("apache/submarine:agent-%s", submarine.Spec.Version)
		}
	}
	// pull secrets
	pullSecrets := util.GetSubmarineCommonImage(submarine).PullSecrets
	if pullSecrets != nil {
		deployment.Spec.Template.Spec.ImagePullSecrets = r.CreatePullSecrets(&pullSecrets)
	}

	return deployment
}

// createSubmarineAgent is a function to create submarine-agent.
// Reference: https://github.com/apache/submarine/blob/master/submarine-cloud-v3/artifacts/submarine-agent.yaml
func (r *SubmarineReconciler) createSubmarineAgent(ctx context.Context, submarine *submarineapacheorgv1alpha1.Submarine) error {
	r.Log.Info("Enter createSubmarineAgent")

	// Step1: Create Deployment
	deployment := &appsv1.Deployment{}
	err := r.Get(ctx, types.NamespacedName{Name: agentName, Namespace: submarine.Namespace}, deployment)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		deployment = r.newSubmarineAgentDeployment(ctx, submarine)
		err = r.Create(ctx, deployment)
		r.Log.Info("Create Deployment", "name", deployment.Name)
	} else {
		newDeployment := r.newSubmarineAgentDeployment(ctx, submarine)
		// compare if there are same
		if !r.CompareAgentDeployment(deployment, newDeployment) {
			// update meta with uid
			newDeployment.ObjectMeta = deployment.ObjectMeta
			err = r.Update(ctx, newDeployment)
			r.Log.Info("Update Deployment", "name", deployment.Name)
		}
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

	return nil
}

// CompareAgentDeployment will determine if two Deployments are equal
func (r *SubmarineReconciler) CompareAgentDeployment(oldDeployment, newDeployment *appsv1.Deployment) bool {
	// spec.replicas
	if *oldDeployment.Spec.Replicas != *newDeployment.Spec.Replicas {
		return false
	}
	if len(oldDeployment.Spec.Template.Spec.Containers) != 1 {
		return false
	}
	// spec.template.spec.containers[0].env
	if !util.CompareEnv(oldDeployment.Spec.Template.Spec.Containers[0].Env, newDeployment.Spec.Template.Spec.Containers[0].Env) {
		return false
	}
	// spec.template.spec.containers[0].image
	if oldDeployment.Spec.Template.Spec.Containers[0].Image != newDeployment.Spec.Template.Spec.Containers[0].Image {
		return false
	}
	// spec.template.spec.imagePullSecrets
	if !util.ComparePullSecrets(oldDeployment.Spec.Template.Spec.ImagePullSecrets, newDeployment.Spec.Template.Spec.ImagePullSecrets) {
		return false
	}
	return true
}
