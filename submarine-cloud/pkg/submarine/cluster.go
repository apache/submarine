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
package submarine

import v1 "github.com/apache/submarine/submarine-cloud/pkg/apis/submarine/v1alpha1"

// Cluster represents a Submarine Cluster
type Cluster struct {
	Name           string
	Namespace      string
	Nodes          map[string]*Node
	Status         v1.ClusterStatus
	NodesPlacement v1.NodesPlacementInfo
	ActionsInfo    ClusterActionsInfo
}

// ClusterActionsInfo use to store information about current action on the Cluster
type ClusterActionsInfo struct {
	NbslotsToMigrate int32
}

// GetNodeByID returns a Cluster Node by its ID
// if not present in the cluster return an error
func (c *Cluster) GetNodeByID(id string) (*Node, error) {
	if n, ok := c.Nodes[id]; ok {
		return n, nil
	}
	return nil, nodeNotFoundedError
}
