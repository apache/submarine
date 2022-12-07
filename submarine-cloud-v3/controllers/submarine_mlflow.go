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

func (r *SubmarineReconciler) newSubmarineMlflowPersistentVolumeClaim(ctx context.Context, submarine *submarineapacheorgv1alpha1.Submarine) *corev1.PersistentVolumeClaim {
	pvc, err := util.ParsePersistentVolumeClaimYaml(mlflowYamlPath)
	if err != nil {
		r.Log.Error(err, "ParsePersistentVolumeClaimYaml")
	}
	pvc.Namespace = submarine.Namespace
	err = controllerutil.SetControllerReference(submarine, pvc, r.Scheme)
	if err != nil {
		r.Log.Error(err, "Set PersistentVolumeClaim ControllerReference")
	}
	return pvc
}

func (r *SubmarineReconciler) newSubmarineMlflowDeployment(ctx context.Context, submarine *submarineapacheorgv1alpha1.Submarine) *appsv1.Deployment {
	deployment, err := util.ParseDeploymentYaml(mlflowYamlPath)
	if err != nil {
		r.Log.Error(err, "ParseDeploymentYaml")
	}
	deployment.Namespace = submarine.Namespace
	err = controllerutil.SetControllerReference(submarine, deployment, r.Scheme)
	if err != nil {
		r.Log.Error(err, "Set Deployment ControllerReference")
	}

	// mlflow image
	mlflowImage := submarine.Spec.Mlflow.Image
	if mlflowImage != "" {
		deployment.Spec.Template.Spec.Containers[0].Image = mlflowImage
	} else {
		deployment.Spec.Template.Spec.Containers[0].Image = fmt.Sprintf("apache/submarine:mlflow-%s", submarine.Spec.Version)
	}
	commonImage := util.GetSubmarineCommonImage(submarine)
	// busybox image
	busyboxImage := commonImage.BusyboxImage
	if busyboxImage != "" {
		deployment.Spec.Template.Spec.InitContainers[0].Image = busyboxImage
	}
	// minio/mc image
	mcImage := commonImage.McImage
	if mcImage != "" {
		deployment.Spec.Template.Spec.InitContainers[1].Image = mcImage
	}
	// pull secrets
	pullSecrets := commonImage.PullSecrets
	if pullSecrets != nil {
		deployment.Spec.Template.Spec.ImagePullSecrets = r.CreatePullSecrets(&pullSecrets)
	}

	// If support istio in openshift, we need to add securityContext to pod in to avoid traffic error
	if r.ClusterType == "openshift" && r.SeldonIstioEnable {
		initcontainers := deployment.Spec.Template.Spec.InitContainers
		for i := range initcontainers {
			initcontainers[i].SecurityContext = util.CreateIstioSidecarSecurityContext(istioSidecarUid)
		}
	}

	return deployment
}

func (r *SubmarineReconciler) newSubmarineMlflowService(ctx context.Context, submarine *submarineapacheorgv1alpha1.Submarine) *corev1.Service {
	service, err := util.ParseServiceYaml(mlflowYamlPath)
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

// createSubmarineMlflow is a function to create submarine-mlflow.
// Reference: https://github.com/apache/submarine/blob/master/submarine-cloud-v3/artifacts/submarine-mlflow.yaml
func (r *SubmarineReconciler) createSubmarineMlflow(ctx context.Context, submarine *submarineapacheorgv1alpha1.Submarine) error {
	r.Log.Info("Enter createSubmarineMlflow")

	// Step 1: Create PersistentVolumeClaim
	pvc := &corev1.PersistentVolumeClaim{}
	err := r.Get(ctx, types.NamespacedName{Name: mlflowPvcName, Namespace: submarine.Namespace}, pvc)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		pvc = r.newSubmarineMlflowPersistentVolumeClaim(ctx, submarine)
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

	// Step 2: Create Deployment
	deployment := &appsv1.Deployment{}
	err = r.Get(ctx, types.NamespacedName{Name: mlflowName, Namespace: submarine.Namespace}, deployment)
	if errors.IsNotFound(err) {
		deployment = r.newSubmarineMlflowDeployment(ctx, submarine)
		err = r.Create(ctx, deployment)
		r.Log.Info("Create Deployment", "name", deployment.Name)
	} else {
		newDeployment := r.newSubmarineMlflowDeployment(ctx, submarine)
		// compare if there are same
		if !r.CompareMlflowDeployment(deployment, newDeployment) {
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
		return err
	}

	if !metav1.IsControlledBy(deployment, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, deployment.Name)
		r.Recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return fmt.Errorf(msg)
	}

	// Step 3: Create Service
	service := &corev1.Service{}
	err = r.Get(ctx, types.NamespacedName{Name: mlflowServiceName, Namespace: submarine.Namespace}, service)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		// If an error occurs during Get/Create, we'll requeue the item so we can
		// attempt processing again later. This could have been caused by a
		// temporary network failure, or any other transient reason.
		service = r.newSubmarineMlflowService(ctx, submarine)
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

// CompareMlflowDeployment will determine if two Deployments are equal
func (r *SubmarineReconciler) CompareMlflowDeployment(oldDeployment, newDeployment *appsv1.Deployment) bool {
	// spec.replicas
	if *oldDeployment.Spec.Replicas != *newDeployment.Spec.Replicas {
		return false
	}

	if len(oldDeployment.Spec.Template.Spec.Containers) != 1 {
		return false
	}
	// spec.template.spec.containers[0].env
	if !util.CompareEnv(oldDeployment.Spec.Template.Spec.Containers[0].Env,
		newDeployment.Spec.Template.Spec.Containers[0].Env) {
		return false
	}
	// spec.template.spec.containers[0].image
	if oldDeployment.Spec.Template.Spec.Containers[0].Image !=
		newDeployment.Spec.Template.Spec.Containers[0].Image {
		return false
	}

	if len(oldDeployment.Spec.Template.Spec.InitContainers) != 2 || len(newDeployment.Spec.Template.Spec.InitContainers) != 2 {
		return false
	}
	for index, old := range oldDeployment.Spec.Template.Spec.InitContainers {
		container := newDeployment.Spec.Template.Spec.InitContainers[index]
		// spec.template.spec.initContainers.image
		if old.Image != container.Image {
			return false
		}
		// spec.template.spec.initContainers.command
		if !util.CompareSlice(old.Command, container.Command) {
			return false
		}
		// spec.template.spec.initContainers.SecurityContext
		if r.ClusterType == "openshift" && r.SeldonIstioEnable {
			sc := old.SecurityContext
			if sc == nil {
				return false
			} else {
				if util.CompareInt64(sc.RunAsUser, container.SecurityContext.RunAsUser) {
					return false
				}
				if util.CompareInt64(sc.RunAsGroup, container.SecurityContext.RunAsGroup) {
					return false
				}
			}
		}
	}

	// spec.template.spec.imagePullSecrets
	if !util.ComparePullSecrets(oldDeployment.Spec.Template.Spec.ImagePullSecrets,
		newDeployment.Spec.Template.Spec.ImagePullSecrets) {
		return false
	}

	return true
}
