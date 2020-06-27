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
	"github.com/golang/glog"
	kapiv1 "k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/record"
)

// ServicesControlInterface interface for the ServicesControl
type ServicesControlInterface interface {
	// CreateSubmarineClusterService used to create the Kubernetes Service needed to access the Submarine Cluster
	CreateSubmarineClusterService(submarineCluster *rapi.SubmarineCluster) (*kapiv1.Service, error)
	// DeleteSubmarineClusterService used to delete the Kubernetes Service linked to the Submarine Cluster
	DeleteSubmarineClusterService(submarineCluster *rapi.SubmarineCluster) error
	// GetSubmarineClusterService used to retrieve the Kubernetes Service associated to the SubmarineCluster
	GetSubmarineClusterService(submarineCluster *rapi.SubmarineCluster) (*kapiv1.Service, error)
}

// ServicesControl contains all information for managing Kube Services
type ServicesControl struct {
	KubeClient clientset.Interface
	Recorder   record.EventRecorder
}

// NewServicesControl builds and returns new ServicesControl instance
func NewServicesControl(client clientset.Interface, rec record.EventRecorder) *ServicesControl {
	glog.Infof("NewServicesControl()")
	ctrl := &ServicesControl{
		KubeClient: client,
		Recorder:   rec,
	}

	return ctrl
}

// GetSubmarineClusterService used to retrieve the Kubernetes Service associated to the SubmarineCluster
func (s *ServicesControl) GetSubmarineClusterService(submarineCluster *rapi.SubmarineCluster) (*kapiv1.Service, error) {
	glog.Infof("GetSubmarineClusterService()")
	return nil, nil
}

// CreateSubmarineClusterService used to create the Kubernetes Service needed to access the Submarine Cluster
func (s *ServicesControl) CreateSubmarineClusterService(submarineCluster *rapi.SubmarineCluster) (*kapiv1.Service, error) {
	glog.Infof("CreateSubmarineClusterService()")
	return nil, nil
}

// DeleteSubmarineClusterService used to delete the Kubernetes Service linked to the Submarine Cluster
func (s *ServicesControl) DeleteSubmarineClusterService(submarineCluster *rapi.SubmarineCluster) error {
	glog.Infof("DeleteSubmarineClusterService()")
	return nil
}

func getServiceName(submarineCluster *rapi.SubmarineCluster) string {
	serviceName := submarineCluster.Name
	if submarineCluster.Spec.ServiceName != "" {
		serviceName = submarineCluster.Spec.ServiceName
	}
	return serviceName
}
