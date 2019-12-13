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
	"github.com/golang/glog"
	kapiv1 "k8s.io/api/core/v1"
)

// IsSubmarineClusterDefaulted check if the SubmarineCluster is already defaulted
func IsSubmarineClusterDefaulted(rc *SubmarineCluster) bool {
	if rc.Spec.NumberOfMaster == nil {
		return false
	}
	if rc.Spec.ReplicationFactor == nil {
		return false
	}
	return true
}

// DefaultSubmarineCluster defaults SubmarineCluster
func DefaultSubmarineCluster(undefaultSubmarineCluster *SubmarineCluster) *SubmarineCluster {
	glog.Infof("DefaultSubmarineCluster()")
	rc := undefaultSubmarineCluster.DeepCopy()
	if rc.Spec.NumberOfMaster == nil {
		rc.Spec.NumberOfMaster = NewInt32(3)
	}
	if rc.Spec.ReplicationFactor == nil {
		rc.Spec.ReplicationFactor = NewInt32(1)
	}

	if rc.Spec.PodTemplate == nil {
		rc.Spec.PodTemplate = &kapiv1.PodTemplateSpec{}
	}

	rc.Status.Cluster.NumberOfMaster = 0
	rc.Status.Cluster.MinReplicationFactor = 0
	rc.Status.Cluster.MaxReplicationFactor = 0
	rc.Status.Cluster.NbPods = 0
	rc.Status.Cluster.NbPodsReady = 0
	rc.Status.Cluster.NbSubmarineRunning = 0

	return rc
}

// NewInt32 use to instanciate a int32 pointer
func NewInt32(val int32) *int32 {
	output := new(int32)
	*output = val

	return output
}
