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

	submarineapacheorgv1 "github.com/apache/submarine/submarine-cloud-v3/api/v1"

	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

func (r *SubmarineReconciler) newSubmarineMinioPersistentVolumeClaim(ctx context.Context, submarine *submarineapacheorgv1.Submarine) *corev1.PersistentVolumeClaim {
	pvc, err := util.ParsePersistentVolumeClaimYaml(minioYamlPath)
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

func (r *SubmarineReconciler) newSubmarineMinioDeployment(ctx context.Context, submarine *submarineapacheorgv1.Submarine) *appsv1.Deployment {
	deployment, err := util.ParseDeploymentYaml(minioYamlPath)
	if err != nil {
		r.Log.Error(err, "ParseDeploymentYaml")
	}
	deployment.Namespace = submarine.Namespace
	err = controllerutil.SetControllerReference(submarine, deployment, r.Scheme)
	if err != nil {
		r.Log.Error(err, "Set Deployment ControllerReference")
	}

	// minio/mc image
	minoImage := submarine.Spec.Minio.Image
	if minoImage != "" {
		deployment.Spec.Template.Spec.Containers[0].Image = minoImage
	}
	// pull secrets
	pullSecrets := util.GetSubmarineCommonImage(submarine).PullSecrets
	if pullSecrets != nil {
		deployment.Spec.Template.Spec.ImagePullSecrets = r.CreatePullSecrets(&pullSecrets)
	}

	return deployment
}

func (r *SubmarineReconciler) newSubmarineMinioService(ctx context.Context, submarine *submarineapacheorgv1.Submarine) *corev1.Service {
	service, err := util.ParseServiceYaml(minioYamlPath)
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

// newSubmarineSeldonSecret is a function to create seldon secret which stores minio connection configurations
func (r *SubmarineReconciler) newSubmarineMinoSecret(ctx context.Context, submarine *submarineapacheorgv1.Submarine) *corev1.Secret {
	secret, err := util.ParseSecretYaml(minioYamlPath)
	if err != nil {
		r.Log.Error(err, "ParseSecretYaml")
	}
	secret.Namespace = submarine.Namespace
	err = controllerutil.SetControllerReference(submarine, secret, r.Scheme)
	if err != nil {
		r.Log.Error(err, "Set Secret ControllerReference")
	}

	// access_ey and secret_key
	accessKey := submarine.Spec.Minio.AccessKey
	if accessKey != "" {
		secret.StringData["MINIO_ACCESS_KEY"] = accessKey
	}
	secretKey := submarine.Spec.Minio.SecretKey
	if secretKey != "" {
		secret.StringData["MINIO_SECRET_KEY"] = secretKey
	}

	return secret
}

// createSubmarineMinio is a function to create submarine-minio.
// Reference: https://github.com/apache/submarine/blob/master/submarine-cloud-v3/artifacts/submarine-minio.yaml
func (r *SubmarineReconciler) createSubmarineMinio(ctx context.Context, submarine *submarineapacheorgv1.Submarine) error {
	r.Log.Info("Enter createSubmarineMinio")

	// Step 1: Create PersistentVolumeClaim
	pvc := &corev1.PersistentVolumeClaim{}
	err := r.Get(ctx, types.NamespacedName{Name: minioPvcName, Namespace: submarine.Namespace}, pvc)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		pvc = r.newSubmarineMinioPersistentVolumeClaim(ctx, submarine)
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

	// Step 2: Create Minio Secret
	secret := &corev1.Secret{}
	err = r.Get(ctx, types.NamespacedName{Name: "submarine-minio-secret", Namespace: submarine.Namespace}, secret)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		secret = r.newSubmarineMinoSecret(ctx, submarine)
		err = r.Create(ctx, secret)
		r.Log.Info("Create Minio Secret", "name", secret.Name)
	} else {
		newSecret := r.newSubmarineMinoSecret(ctx, submarine)
		// compare if there are same
		if !util.CompareSecret(secret, newSecret) {
			// update meta with uid
			newSecret.ObjectMeta = secret.ObjectMeta
			err = r.Update(ctx, newSecret)
			r.Log.Info("Update Minio Secret", "name", secret.Name)
		}
	}

	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if err != nil {
		return err
	}

	if !metav1.IsControlledBy(secret, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, secret.Name)
		r.Recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return fmt.Errorf(msg)
	}

	// Step 3: Create Deployment
	deployment := &appsv1.Deployment{}
	err = r.Get(ctx, types.NamespacedName{Name: minioName, Namespace: submarine.Namespace}, deployment)
	if errors.IsNotFound(err) {
		deployment = r.newSubmarineMinioDeployment(ctx, submarine)
		err = r.Create(ctx, deployment)
		r.Log.Info("Create Deployment", "name", deployment.Name)
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

	// Step 4: Create Service
	service := &corev1.Service{}
	err = r.Get(ctx, types.NamespacedName{Name: minioServiceName, Namespace: submarine.Namespace}, service)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		service = r.newSubmarineMinioService(ctx, submarine)
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
