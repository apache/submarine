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

	istiov1alpha3 "istio.io/client-go/pkg/apis/networking/v1alpha3"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	submarineapacheorgv1alpha1 "github.com/apache/submarine/submarine-cloud-v3/api/v1alpha1"

	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

func (r *SubmarineReconciler) newSubmarineVirtualService(ctx context.Context, submarine *submarineapacheorgv1alpha1.Submarine) *istiov1alpha3.VirtualService {
	virtualService, err := ParseVirtualService(virtualServiceYamlPath)
	if err != nil {
		r.Log.Error(err, "ParseVirtualService")
	}
	virtualService.Namespace = submarine.Namespace

	// Virtualservice is optional
	// so that if user makes a declaration, we need to modify the configuration
	specVirtual := submarine.Spec.Virtualservice
	if specVirtual != nil {
		virtualserviceHosts := specVirtual.Hosts
		if virtualserviceHosts != nil {
			// Use `Hosts` defined in submarine spec
			virtualService.Spec.Hosts = virtualserviceHosts
		}
		virtualserviceGateways := specVirtual.Gateways
		if virtualserviceGateways != nil {
			// Use `Gateways` defined in submarine spec
			virtualService.Spec.Gateways = virtualserviceGateways
		}
	}

	err = controllerutil.SetControllerReference(submarine, virtualService, r.Scheme)
	if err != nil {
		r.Log.Error(err, "Set VirtualService ControllerReference")
	}
	return virtualService
}

// createVirtualService is a function to create VirtualService.
// Reference: https://github.com/apache/submarine/blob/master/submarine-cloud-v3/artifacts/submarine-virtualservice.yaml
func (r *SubmarineReconciler) createVirtualService(ctx context.Context, submarine *submarineapacheorgv1alpha1.Submarine) error {
	r.Log.Info("Enter createIngress")

	virtualService := &istiov1alpha3.VirtualService{}
	err := r.Get(ctx, types.NamespacedName{Name: virtualServiceName, Namespace: submarine.Namespace}, virtualService)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		virtualService = r.newSubmarineVirtualService(ctx, submarine)
		err = r.Create(ctx, virtualService)
		r.Log.Info("Create VirtualService", "name", virtualService.Name)
	}

	// If an error occurs during Get/Create, we'll requeue the item so we can
	// attempt processing again later. This could have been caused by a
	// temporary network failure, or any other transient reason.
	if err != nil {
		return err
	}

	if !metav1.IsControlledBy(virtualService, submarine) {
		msg := fmt.Sprintf(MessageResourceExists, virtualService.Name)
		r.Recorder.Event(submarine, corev1.EventTypeWarning, ErrResourceExists, msg)
		return fmt.Errorf(msg)
	}

	return nil
}
