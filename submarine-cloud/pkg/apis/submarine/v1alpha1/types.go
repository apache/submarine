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
	"fmt"
	kapiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// SubmarineCluster represents a Submarine Cluster
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type SubmarineCluster struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// Spec represents the desired SubmarineCluster specification
	Spec SubmarineClusterSpec `json:"spec,omitempty"`

	// Status represents the current SubmarineCluster status
	Status SubmarineClusterStatus `json:"status,omitempty"`
}

// SubmarineClusterList is a list of Submarine resources
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type SubmarineClusterList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	Items []SubmarineCluster `json:"items"`
}

// SubmarineClusterSpec contains SubmarineCluster specification
type SubmarineClusterSpec struct {
	NumberOfMaster    *int32 `json:"numberOfMaster,omitempty"`
	ReplicationFactor *int32 `json:"replicationFactor,omitempty"`

	// ServiceName name used to create the Kubernetes Service that reference the Submarine Cluster nodes.
	// if ServiceName is empty, the SubmarineCluster.Name will be use for creating the service.
	ServiceName string `json:"serviceName,omitempty"`

	// PodTemplate contains the pod specification that should run the Submarine-server process
	PodTemplate *kapiv1.PodTemplateSpec `json:"podTemplate,omitempty"`

	// Labels for created Submarine-cluster (deployment, rs, pod) (if any)
	AdditionalLabels map[string]string `json:"AdditionalLabels,omitempty"`

	Name    string `json:"name"`
	School  string `json:"school"`
	Email   string `json:"email"`
	Address string `json:"address"`
}

// SubmarineClusterNode represent a SubmarineCluster Node
type SubmarineClusterNode struct {
	ID        string                   `json:"id"`
	Role      SubmarineClusterNodeRole `json:"role"`
	IP        string                   `json:"ip"`
	Port      string                   `json:"port"`
	Slots     []string                 `json:"slots,omitempty"`
	MasterRef string                   `json:"masterRef,omitempty"`
	PodName   string                   `json:"podName"`
	Pod       *kapiv1.Pod              `json:"-"`
}

func (n SubmarineClusterNode) String() string {
	if n.Role != SubmarineClusterNodeRoleSlave {
		return fmt.Sprintf("(Master:%s, Addr:%s:%s, PodName:%s, Slots:%v)", n.ID, n.IP, n.Port, n.PodName, n.Slots)
	}
	return fmt.Sprintf("(Slave:%s, Addr:%s:%s, PodName:%s, MasterRef:%s)", n.ID, n.IP, n.Port, n.PodName, n.MasterRef)
}

// SubmarineClusterConditionType is the type of SubmarineClusterCondition
type SubmarineClusterConditionType string

const (
	// SubmarineClusterOK means the SubmarineCluster is in a good shape
	SubmarineClusterOK SubmarineClusterConditionType = "ClusterOK"

	// SubmarineClusterScaling means the SubmarineCluster is currently in a scaling stage
	SubmarineClusterScaling SubmarineClusterConditionType = "Scaling"

	// SubmarineClusterRebalancing means the SubmarineCluster is currenlty rebalancing slots and keys
	SubmarineClusterRebalancing SubmarineClusterConditionType = "Rebalancing"

	// SubmarineClusterRollingUpdate means the SubmarineCluster is currenlty performing a rolling update of its nodes
	SubmarineClusterRollingUpdate SubmarineClusterConditionType = "RollingUpdate"
)

// SubmarineClusterNodeRole SubmarineCluster Node Role type
type SubmarineClusterNodeRole string

const (
	// SubmarineClusterNodeRoleMaster SubmarineCluster Master node role
	SubmarineClusterNodeRoleMaster SubmarineClusterNodeRole = "Master"

	// SubmarineClusterNodeRoleSlave SubmarineCluster Master node role
	SubmarineClusterNodeRoleSlave SubmarineClusterNodeRole = "Slave"

	// SubmarineClusterNodeRoleNone None node role
	SubmarineClusterNodeRoleNone SubmarineClusterNodeRole = "None"
)

// ClusterStatus Submarine Cluster status
type ClusterStatus string

const (
	// ClusterStatusOK ClusterStatus OK
	ClusterStatusOK ClusterStatus = "OK"

	// ClusterStatusError ClusterStatus Error
	ClusterStatusError ClusterStatus = "Error"

	// ClusterStatusScaling ClusterStatus Scaling
	ClusterStatusScaling ClusterStatus = "Scaling"

	// ClusterStatusCalculatingRebalancing ClusterStatus Rebalancing
	ClusterStatusCalculatingRebalancing ClusterStatus = "Calculating Rebalancing"

	// ClusterStatusRebalancing ClusterStatus Rebalancing
	ClusterStatusRebalancing ClusterStatus = "Rebalancing"

	// ClusterStatusRollingUpdate ClusterStatus RollingUpdate
	ClusterStatusRollingUpdate ClusterStatus = "RollingUpdate"
)

// SubmarineClusterStatus contains SubmarineCluster status
type SubmarineClusterStatus struct {
	// Conditions represent the latest available observations of an object's current state.
	Conditions []SubmarineClusterCondition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`
	// Status of the condition, one of True, False, Unknown.
	Status kapiv1.ConditionStatus `json:"status"`
	// StartTime represents time when the workflow was acknowledged by the Workflow controller
	// It is not guaranteed to be set in happens-before order across separate operations.
	// It is represented in RFC3339 form and is in UTC.
	// StartTime doesn't consider start time of `ExternalReference`
	StartTime *metav1.Time `json:"startTime,omitempty"`
	// (brief) reason for the condition's last transition.
	Reason string `json:"reason,omitempty"`
	// Human readable message indicating details about last transition.
	Message string `json:"message,omitempty"`
	// Cluster a view of the current SubmarineCluster
	Cluster SubmarineClusterClusterStatus
}

// SubmarineClusterCondition represent the condition of the SubmarineCluster
type SubmarineClusterCondition struct {
	// Type of workflow condition
	Type SubmarineClusterConditionType `json:"type"`
	// Status of the condition, one of True, False, Unknown.
	Status kapiv1.ConditionStatus `json:"status"`
	// Last time the condition was checked.
	LastProbeTime metav1.Time `json:"lastProbeTime,omitempty"`
	// Last time the condition transited from one status to another.
	LastTransitionTime metav1.Time `json:"lastTransitionTime,omitempty"`
	// (brief) reason for the condition's last transition.
	Reason string `json:"reason,omitempty"`
	// Human readable message indicating details about last transition.
	Message string `json:"message,omitempty"`
}

// SubmarineClusterClusterStatus represent the Submarine Cluster status
type SubmarineClusterClusterStatus struct {
	Status               ClusterStatus `json:"status"`
	NumberOfMaster       int32         `json:"numberOfMaster,omitempty"`
	MinReplicationFactor int32         `json:"minReplicationFactor,omitempty"`
	MaxReplicationFactor int32         `json:"maxReplicationFactor,omitempty"`

	NodesPlacement NodesPlacementInfo `json:"nodesPlacementInfo,omitempty"`

	// In theory, we always have NbPods > NbSubmarineRunning > NbPodsReady
	NbPods             int32 `json:"nbPods,omitempty"`
	NbPodsReady        int32 `json:"nbPodsReady,omitempty"`
	NbSubmarineRunning int32 `json:"nbSubmarineNodesRunning,omitempty"`

	Nodes []SubmarineClusterNode `json:"nodes"`
}

func (s SubmarineClusterClusterStatus) String() string {
	output := ""
	output += fmt.Sprintf("status:%s\n", s.Status)
	output += fmt.Sprintf("NumberOfMaster:%d\n", s.NumberOfMaster)
	output += fmt.Sprintf("MinReplicationFactor:%d\n", s.MinReplicationFactor)
	output += fmt.Sprintf("MaxReplicationFactor:%d\n", s.MaxReplicationFactor)
	output += fmt.Sprintf("NodesPlacement:%s\n\n", s.NodesPlacement)
	output += fmt.Sprintf("NbPods:%d\n", s.NbPods)
	output += fmt.Sprintf("NbPodsReady:%d\n", s.NbPodsReady)
	output += fmt.Sprintf("NbSubmarineRunning:%d\n\n", s.NbSubmarineRunning)

	output += fmt.Sprintf("Nodes (%d): %s\n", len(s.Nodes), s.Nodes)

	return output
}

// NodesPlacementInfo Submarine Nodes placement mode information
type NodesPlacementInfo string

const (
	// NodesPlacementInfoBestEffort the cluster nodes placement is in best effort,
	// it means you can have 2 masters (or more) on the same VM.
	NodesPlacementInfoBestEffort NodesPlacementInfo = "BestEffort"
	// NodesPlacementInfoOptimal the cluster nodes placement is optimal,
	// it means on master by VM
	NodesPlacementInfoOptimal NodesPlacementInfo = "Optimal"
)
