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

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Submarine is a specification for a Submarine resource
type Submarine struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   SubmarineSpec   `json:"spec"`
	Status SubmarineStatus `json:"status"`
}

type SubmarineServer struct {
	Image    string `json:"image"`
	Replicas *int32 `json:"replicas"`
}

type SubmarineDatabase struct {
	Image                   string `json:"image"`
	Replicas                *int32 `json:"replicas"`
	StorageSize             string `json:"storageSize"`
	MysqlRootPasswordSecret string `json:"mysqlRootPasswordSecret"`
}

type SubmarineTensorboard struct {
	Enabled     *bool  `json:"enabled"`
	StorageSize string `json:"storageSize"`
}

type SubmarineMlflow struct {
	Enabled     *bool  `json:"enabled"`
	StorageSize string `json:"storageSize"`
}

type SubmarineStorage struct {
	StorageType string `json:"storageType"`
	HostPath    string `json:"hostPath"`
	NfsPath     string `json:"nfsPath"`
	NfsIP       string `json:"nfsIP"`
}

// SubmarineSpec is the spec for a Submarine resource
type SubmarineSpec struct {
	Version     string                `json:"version"`
	Replicas    *int32                `json:"replicas"`
	Server      *SubmarineServer      `json:"server"`
	Database    *SubmarineDatabase    `json:"database"`
	Tensorboard *SubmarineTensorboard `json:"tensorboard"`
	Mlflow      *SubmarineMlflow      `json:"mlflow"`
	Storage     *SubmarineStorage     `json:"storage"`
}

// SubmarineStatus is the status for a Submarine resource
type SubmarineStatus struct {
	AvailableReplicas int32 `json:"availableReplicas"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// SubmarineList is a list of Submarine resources
type SubmarineList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	Items []Submarine `json:"items"`
}
