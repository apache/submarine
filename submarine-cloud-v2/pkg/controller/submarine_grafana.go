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
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
)

func newSubmarineGrafanaIngressRoute(submarine *v1alpha1.Submarine) *traefikv1alpha1.IngressRoute {
	ingressRoute, err := ParseIngressRouteYaml(grafanaYamlPath)
	if err != nil {
		klog.Info("[Error] ParseIngressRouteYaml", err)
	}
	ingressRoute.ObjectMeta.OwnerReferences = []metav1.OwnerReference{
		*metav1.NewControllerRef(submarine, v1alpha1.SchemeGroupVersion.WithKind("Submarine")),
	}
	return ingressRoute
}

// createSubmarineGrafana is a function to create submarine-grafana.
// Reference: https://github.com/apache/submarine/blob/master/helm-charts/submarine/templates/submarine-grafana.yaml
func (c *Controller) createSubmarineGrafana(submarine *v1alpha1.Submarine) error {
	klog.Info("[createSubmarineGrafana]")

	// Create IngressRoute
	ingressroute, err := c.ingressrouteLister.IngressRoutes(submarine.Namespace).Get(grafanaIngressRouteName)
	// If the resource doesn't exist, we'll create it
	if errors.IsNotFound(err) {
		ingressroute, err = c.traefikclientset.TraefikV1alpha1().IngressRoutes(submarine.Namespace).Create(context.TODO(), newSubmarineGrafanaIngressRoute(submarine), metav1.CreateOptions{})
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
