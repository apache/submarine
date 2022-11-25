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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

//+kubebuilder:object:root=true
//+kubebuilder:subresource:status

// Submarine is the Schema for the submarines API
type Submarine struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   SubmarineSpec   `json:"spec,omitempty"`
	Status SubmarineStatus `json:"status,omitempty"`
}

// SubmarineSpec defines the desired state of Submarine
type SubmarineSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	// Version is the submarine docker image version
	Version string `json:"version"`
	// Server is the spec that defines the submarine server
	Server *SubmarineServerSpec `json:"server"`
	// Database is the spec that defines the submarine database
	Database *SubmarineDatabaseSpec `json:"database"`
	// Virtualservice is the spec that defines the submarine virtualservice
	Virtualservice *SubmarineVirtualserviceSpec `json:"virtualservice,omitempty"`
	// Tensorboard is the spec that defines the submarine tensorboard
	Tensorboard *SubmarineTensorboardSpec `json:"tensorboard"`
	// Mlflow is the spec that defines the submarine mlflow
	Mlflow *SubmarineMlflowSpec `json:"mlflow"`
	// Minio is the spec that defines the submarine minio
	Minio *SubmarineMinioSpec `json:"minio"`
	// Common is the spec that defines some submarine common configurations
	Common *SubmarineCommon `json:"common,omitempty"`
}

// SubmarineServerSpec defines the desired submarine server
type SubmarineServerSpec struct {
	// Image is the submarine server's docker image
	Image string `json:"image,omitempty"`
	// Replicas is the number of submarine server's replica
	// +kubebuilder:validation:Minimum=1
	Replicas *int32 `json:"replicas"`
}

// SubmarineServerSpec defines the desired submarine database
type SubmarineDatabaseSpec struct {
	// Image is the submarine database's docker image
	Image string `json:"image,omitempty"`
	// StorageSize is the storage size of the database
	StorageSize string `json:"storageSize"`
	// MysqlRootPasswordSecret is the mysql root password secret
	MysqlRootPasswordSecret string `json:"mysqlRootPasswordSecret"`
}

// SubmarineVirtualserviceSpec defines the desired submarine virtualservice
type SubmarineVirtualserviceSpec struct {
	// Hosts is the submarine virtualservice's destination hosts
	Hosts []string `json:"hosts,omitempty"`
	// Hosts is the submarine virtualservice's gateways
	Gateways []string `json:"gateways,omitempty"`
}

// SubmarineServerSpec defines the desired submarine tensorboard
type SubmarineTensorboardSpec struct {
	// Image is the submarine tensorboard's docker image
	Image string `json:"image,omitempty"`
	// Enabled defines whether to enable tensorboard or not
	Enabled *bool `json:"enabled"`
	// StorageSize defines the storage size of tensorboard
	StorageSize string `json:"storageSize"`
}

// SubmarineServerSpec defines the desired submarine mlflow
type SubmarineMlflowSpec struct {
	// Image is the submarine mlflow's docker image
	Image string `json:"image,omitempty"`
	// Enabled defines whether to enable mlflow or not
	Enabled *bool `json:"enabled"`
	// StorageSize defines the storage size of mlflow
	StorageSize string `json:"storageSize"`
}

// SubmarineServerSpec defines the desired submarine minio
type SubmarineMinioSpec struct {
	// Image is the submarine minio's docker image
	Image string `json:"image,omitempty"`
	// Enabled defines whether to enable minio or not
	Enabled *bool `json:"enabled"`
	// StorageSize defines the storage size of minio
	StorageSize string `json:"storageSize"`
}

// SubmarineCommon defines the observed common configuration
type SubmarineCommon struct {
	// Image is the basic image used in initcontainer in some submarine resources
	Image CommonImage `json:"image"`
}

// CommonImage defines the observed common image info
type CommonImage struct {
	// McImage is the image used by the submarine service resource (server...) to check whether the minio is started in the initcontainer
	McImage string `json:"mc,omitempty"`
	// BusyboxImage is the image used by the submarine service resource (server...) to check whether the minio is started in the initcontainer
	BusyboxImage string `json:"busybox,omitempty"`
	// PullSecrets is a specified array of imagePullSecrets. This secrets must be manually created in the namespace.
	PullSecrets []string `json:"pullSecrets,omitempty"`
}

// SubmarineStatus defines the observed state of Submarine
type SubmarineStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	// AvailableServerReplicas is the current available replicas of submarine server
	AvailableServerReplicas int32 `json:"availableServerReplicas"`
	// AvailableServerReplicas is the current available replicas of submarine database
	AvailableDatabaseReplicas int32 `json:"availableDatabaseReplicas"`
	// SubmarineState tells the overall submarine state.
	SubmarineState `json:"submarineState,omitempty"`
}

// SubmarineStateType represents the type of the current state of a submarine.
type SubmarineStateType string

// Different states a submarine resource may be
const (
	NewState      SubmarineStateType = ""
	CreatingState SubmarineStateType = "CREATING"
	RunningState  SubmarineStateType = "RUNNING"
	FailedState   SubmarineStateType = "FAILED"
)

// SubmarineState tells the current state of the submarine and an error message in case of failures.
type SubmarineState struct {
	State        SubmarineStateType `json:"state"`
	ErrorMessage string             `json:"errorMessage,omitempty"`
}

// +kubebuilder:object:root=true
// SubmarineList contains a list of Submarine
type SubmarineList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []Submarine `json:"items"`
}

func init() {
	SchemeBuilder.Register(&Submarine{}, &SubmarineList{})
}
