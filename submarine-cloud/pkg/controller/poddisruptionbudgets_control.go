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
	rapi "github.com/apache/submarine/submarine-cloud/pkg/apis/submarine/v1alpha1"
	policyv1 "k8s.io/api/policy/v1beta1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/record"
)

// PodDisruptionBudgetsControlInterface inferface for the PodDisruptionBudgetsControl
type PodDisruptionBudgetsControlInterface interface {
	// CreateSubmarineClusterPodDisruptionBudget used to create the Kubernetes PodDisruptionBudget needed to access the Submarine Cluster
	CreateSubmarineClusterPodDisruptionBudget(submarineCluster *rapi.SubmarineCluster) (*policyv1.PodDisruptionBudget, error)
	// DeleteSubmarineClusterPodDisruptionBudget used to delete the Kubernetes PodDisruptionBudget linked to the Submarine Cluster
	DeleteSubmarineClusterPodDisruptionBudget(submarineCluster *rapi.SubmarineCluster) error
	// GetSubmarineClusterPodDisruptionBudget used to retrieve the Kubernetes PodDisruptionBudget associated to the SubmarineCluster
	GetSubmarineClusterPodDisruptionBudget(submarineCluster *rapi.SubmarineCluster) (*policyv1.PodDisruptionBudget, error)
}

// PodDisruptionBudgetsControl contains all information for managing Kube PodDisruptionBudgets
type PodDisruptionBudgetsControl struct {
	KubeClient clientset.Interface
	Recorder   record.EventRecorder
}

// NewPodDisruptionBudgetsControl builds and returns new PodDisruptionBudgetsControl instance
func NewPodDisruptionBudgetsControl(client clientset.Interface, rec record.EventRecorder) *PodDisruptionBudgetsControl {
	ctrl := &PodDisruptionBudgetsControl{
		KubeClient: client,
		Recorder:   rec,
	}

	return ctrl
}

// GetSubmarineClusterPodDisruptionBudget used to retrieve the Kubernetes PodDisruptionBudget associated to the SubmarineCluster
func (s *PodDisruptionBudgetsControl) GetSubmarineClusterPodDisruptionBudget(submarineCluster *rapi.SubmarineCluster) (*policyv1.PodDisruptionBudget, error) {
	return nil, nil
}

// DeleteSubmarineClusterPodDisruptionBudget used to delete the Kubernetes PodDisruptionBudget linked to the Submarine Cluster
func (s *PodDisruptionBudgetsControl) DeleteSubmarineClusterPodDisruptionBudget(submarineCluster *rapi.SubmarineCluster) error {
	return nil
}

// CreateSubmarineClusterPodDisruptionBudget used to create the Kubernetes PodDisruptionBudget needed to access the Submarine Cluster
func (s *PodDisruptionBudgetsControl) CreateSubmarineClusterPodDisruptionBudget(submarineCluster *rapi.SubmarineCluster) (*policyv1.PodDisruptionBudget, error) {

	return nil, nil
}
