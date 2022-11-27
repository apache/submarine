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
	submarineapacheorgv1alpha1 "github.com/apache/submarine/submarine-cloud-v3/api/v1alpha1"
	"github.com/apache/submarine/submarine-cloud-v3/controllers/util"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

// newSubmarineSeldonSecret is a function to create seldon secret which stores minio connection configurations
func (r *SubmarineReconciler) newSubmarineSeldonSecret(ctx context.Context, submarine *submarineapacheorgv1alpha1.Submarine) *corev1.Secret {
	secret, err := util.ParseSecretYaml(serveYamlPath)
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
		secret.StringData["RCLONE_CONFIG_S3_ACCESS_KEY_ID"] = accessKey
	}
	secretKey := submarine.Spec.Minio.SecretKey
	if secretKey != "" {
		secret.StringData["RCLONE_CONFIG_S3_SECRET_ACCESS_KEY"] = secretKey
	}

	return secret
}

// createSubmarineServe is a function to create submarine-serve.
// Reference: https://github.com/apache/submarine/blob/master/submarine-cloud-v3/artifacts/submarine-serve.yaml
func (r *SubmarineReconciler) createSubmarineServe(ctx context.Context, submarine *submarineapacheorgv1alpha1.Submarine) error {
	r.Log.Info("Enter createSubmarineServe")

	// Step 1: Create Seldon Secret
	secret := &corev1.Secret{}
	err := r.Get(ctx, types.NamespacedName{Name: "submarine-serve-secret", Namespace: submarine.Namespace}, secret)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		secret = r.newSubmarineSeldonSecret(ctx, submarine)
		err = r.Create(ctx, secret)
		r.Log.Info("Create Seldon Secret", "name", secret.Name)
	} else {
		newSecret := r.newSubmarineSeldonSecret(ctx, submarine)
		// compare if there are same
		if !util.CompareSecret(secret, newSecret) {
			// update meta with uid
			newSecret.ObjectMeta = secret.ObjectMeta
			err = r.Update(ctx, secret)
			r.Log.Info("Update Seldon Secret", "name", secret.Name)
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

	return nil
}
