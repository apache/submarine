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
package clustering

import (
	"github.com/apache/submarine/submarine-cloud/pkg/submarine"
	"github.com/golang/glog"
)

// AttachingSlavesToMaster used to attach slaves to there masters
func AttachingSlavesToMaster(cluster *submarine.Cluster, admin submarine.AdminInterface, slavesByMaster map[string]submarine.Nodes) error {
	var globalErr error
	for masterID, slaves := range slavesByMaster {
		masterNode, err := cluster.GetNodeByID(masterID)
		if err != nil {
			glog.Errorf("[AttachingSlavesToMaster] unable fo found the Cluster.Node with submarine ID:%s", masterID)
			continue
		}
		for _, slave := range slaves {
			glog.V(2).Infof("[AttachingSlavesToMaster] Attaching node %s to master %s", slave.ID, masterID)

			err := admin.AttachSlaveToMaster(slave, masterNode)
			if err != nil {
				glog.Errorf("Error while attaching node %s to master %s: %v", slave.ID, masterID, err)
				globalErr = err
			}
		}
	}
	return globalErr
}
