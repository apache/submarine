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
package sanitycheck

import (
	rapi "github.com/apache/submarine/submarine-cloud/pkg/apis/submarine/v1alpha1"
	"github.com/apache/submarine/submarine-cloud/pkg/config"
	"github.com/apache/submarine/submarine-cloud/pkg/controller/pod"
	"github.com/apache/submarine/submarine-cloud/pkg/submarine"
)

// RunSanityChecks function used to run all the sanity check on the current cluster
// Return actionDone = true if a modification has been made on the cluster
func RunSanityChecks(admin submarine.AdminInterface, config *config.Submarine, podControl pod.SubmarineClusterControlInterface, cluster *rapi.SubmarineCluster, infos *submarine.ClusterInfos, dryRun bool) (actionDone bool, err error) {
	/*
		// * fix failed nodes: in some cases (cluster without enough master after crash or scale down), some nodes may still know about fail nodes
		if actionDone, err = FixFailedNodes(admin, cluster, infos, dryRun); err != nil {
			return actionDone, err
		} else if actionDone {
			glog.V(2).Infof("FixFailedNodes done an action on the cluster (dryRun:%v)", dryRun)
			return actionDone, nil
		}

		// forget nodes and delete pods when a submarine node is untrusted.
		if actionDone, err = FixUntrustedNodes(admin, podControl, cluster, infos, dryRun); err != nil {
			return actionDone, err
		} else if actionDone {
			glog.V(2).Infof("FixUntrustedNodes done an action on the cluster (dryRun:%v)", dryRun)
			return actionDone, nil
		}

		// forget nodes and delete pods when a submarine node is untrusted.
		if actionDone, err = FixTerminatingPods(cluster, podControl, 5*time.Minute, dryRun); err != nil {
			return actionDone, err
		} else if actionDone {
			glog.V(2).Infof("FixTerminatingPods done an action on the cluster (dryRun:%v)", dryRun)
			return actionDone, nil
		}

		// forget nodes and delete pods when a submarine node is untrusted.
		if actionDone, err = FixClusterSplit(admin, config, infos, dryRun); err != nil {
			return actionDone, err
		} else if actionDone {
			glog.V(2).Infof("FixClusterSplit done an action on the cluster (dryRun:%v)", dryRun)
			return actionDone, nil
		}*/

	return true, nil ///actionDone, err
}
