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
	"github.com/apache/submarine/submarine-cloud/pkg/controller/clustering"
	"github.com/apache/submarine/submarine-cloud/pkg/submarine"
	"github.com/golang/glog"
	"time"
)

// Perform various management operations on Submarine Pod and Submarine clusters to approximate the desired state
func (c *Controller) clusterAction(admin submarine.AdminInterface, cluster *rapi.SubmarineCluster, infos *submarine.ClusterInfos) (bool, error) {
	glog.Info("clusterAction()")
	var err error
	/* run sanity check if needed
	needSanity, err := sanitycheck.RunSanityChecks(admin, &c.config.submarine, c.podControl, cluster, infos, true)
	if err != nil {
		glog.Errorf("[clusterAction] cluster %s/%s, an error occurs during sanitycheck: %v ", cluster.Namespace, cluster.Name, err)
		return false, err
	}
	if needSanity {
		glog.V(3).Infof("[clusterAction] run sanitycheck cluster: %s/%s", cluster.Namespace, cluster.Name)
		return sanitycheck.RunSanityChecks(admin, &c.config.submarine, c.podControl, cluster, infos, false)
	}*/

	// Start more pods in needed
	if needMorePods(cluster) {
		if setScalingCondition(&cluster.Status, true) {
			if cluster, err = c.updateHandler(cluster); err != nil {
				return false, err
			}
		}
		pod, err2 := c.podControl.CreatePod(cluster)
		if err2 != nil {
			glog.Errorf("[clusterAction] unable to create a pod associated to the SubmarineCluster: %s/%s, err: %v", cluster.Namespace, cluster.Name, err2)
			return false, err2
		}

		glog.V(3).Infof("[clusterAction]create a Pod %s/%s", pod.Namespace, pod.Name)
		return true, nil
	}
	if setScalingCondition(&cluster.Status, false) {
		if cluster, err = c.updateHandler(cluster); err != nil {
			return false, err
		}
	}

	// Reconfigure the Cluster if needed
	hasChanged, err := c.applyConfiguration(admin, cluster)
	if err != nil {
		glog.Errorf("[clusterAction] cluster %s/%s, an error occurs: %v ", cluster.Namespace, cluster.Name, err)
		return false, err
	}

	if hasChanged {
		glog.V(6).Infof("[clusterAction] cluster has changed cluster: %s/%s", cluster.Namespace, cluster.Name)
		return true, nil
	}

	glog.Infof("[clusterAction] cluster hasn't changed cluster: %s/%s", cluster.Namespace, cluster.Name)
	return false, nil
}

// applyConfiguration apply new configuration if needed:
// - add or delete pods
// - configure the submarine-server process
func (c *Controller) applyConfiguration(admin submarine.AdminInterface, cluster *rapi.SubmarineCluster) (bool, error) {
	glog.Info("applyConfiguration START")
	defer glog.Info("applyConfiguration STOP")

	asChanged := false

	// expected replication factor and number of master nodes
	cReplicaFactor := *cluster.Spec.ReplicationFactor
	cNbMaster := *cluster.Spec.NumberOfMaster
	// Adapt, convert CR to structure in submarine package
	rCluster, nodes, err := newSubmarineCluster(admin, cluster)
	if err != nil {
		glog.Errorf("Unable to create the SubmarineCluster view, error:%v", err)
		return false, err
	}
	// PodTemplate changes require rolling updates
	if needRollingUpdate(cluster) {
		if setRollingUpdategCondition(&cluster.Status, true) {
			if cluster, err = c.updateHandler(cluster); err != nil {
				return false, err
			}
		}

		glog.Info("applyConfiguration needRollingUpdate")
		return c.manageRollingUpdate(admin, cluster, rCluster, nodes)
	}
	if setRollingUpdategCondition(&cluster.Status, false) {
		if cluster, err = c.updateHandler(cluster); err != nil {
			return false, err
		}
	}

	// if the number of Pods is greater than expected
	if needLessPods(cluster) {
		if setRebalancingCondition(&cluster.Status, true) {
			if cluster, err = c.updateHandler(cluster); err != nil {
				return false, err
			}
		}
		glog.Info("applyConfiguration needLessPods")
		// Configure Submarine cluster
		return c.managePodScaleDown(admin, cluster, rCluster, nodes)
	}
	// If it is not a rolling update, modify the Condition
	if setRebalancingCondition(&cluster.Status, false) {
		if cluster, err = c.updateHandler(cluster); err != nil {
			return false, err
		}
	}

	clusterStatus := &cluster.Status.Cluster
	if (clusterStatus.NbPods - clusterStatus.NbSubmarineRunning) != 0 {
		glog.V(3).Infof("All pods not ready wait to be ready, nbPods: %d, nbPodsReady: %d", clusterStatus.NbPods, clusterStatus.NbSubmarineRunning)
		return false, err
	}

	//First, we define the new masters
	// Select the desired number of Masters and assign Hashslots to each Master. The Master will be distributed to different K8S nodes as much as possible
	// Set the cluster status to Calculating Rebalancing
	newMasters, curMasters, allMaster, err := clustering.DispatchMasters(rCluster, nodes, cNbMaster, admin)
	if err != nil {
		glog.Errorf("Cannot dispatch slots to masters: %v", err)
		rCluster.Status = rapi.ClusterStatusError
		return false, err
	}
	// If the number of new and old masters is not the same
	if len(newMasters) != len(curMasters) {
		asChanged = true
	}

	// Second select Node that is already a slave
	currentSlaveNodes := nodes.FilterByFunc(submarine.IsSlave)

	//New slaves are slaves which is currently a master with no slots
	newSlave := nodes.FilterByFunc(func(nodeA *submarine.Node) bool {
		for _, nodeB := range newMasters {
			if nodeA.ID == nodeB.ID {
				return false
			}
		}
		for _, nodeB := range currentSlaveNodes {
			if nodeA.ID == nodeB.ID {
				return false
			}
		}
		return true
	})

	// Depending on whether we scale up or down, we will dispatch slaves before/after the dispatch of slots
	if cNbMaster < int32(len(curMasters)) {
		// this happens usually after a scale down of the cluster
		// we should dispatch slots before dispatching slaves
		if err := clustering.DispatchSlotToNewMasters(rCluster, admin, newMasters, curMasters, allMaster); err != nil {
			glog.Error("Unable to dispatch slot on new master, err:", err)
			return false, err
		}

		// assign master/slave roles
		newSubmarineSlavesByMaster, bestEffort := clustering.PlaceSlaves(rCluster, newMasters, currentSlaveNodes, newSlave, cReplicaFactor)
		if bestEffort {
			rCluster.NodesPlacement = rapi.NodesPlacementInfoBestEffort
		}

		if err := clustering.AttachingSlavesToMaster(rCluster, admin, newSubmarineSlavesByMaster); err != nil {
			glog.Error("Unable to dispatch slave on new master, err:", err)
			return false, err
		}
	} else {
		// We are scaling up the nbmaster or the nbmaster doesn't change.
		// assign master/slave roles
		newSubmarineSlavesByMaster, bestEffort := clustering.PlaceSlaves(rCluster, newMasters, currentSlaveNodes, newSlave, cReplicaFactor)
		if bestEffort {
			rCluster.NodesPlacement = rapi.NodesPlacementInfoBestEffort
		}

		if err := clustering.AttachingSlavesToMaster(rCluster, admin, newSubmarineSlavesByMaster); err != nil {
			glog.Error("Unable to dispatch slave on new master, err:", err)
			return false, err
		}

		if err := clustering.DispatchSlotToNewMasters(rCluster, admin, newMasters, curMasters, allMaster); err != nil {
			glog.Error("Unable to dispatch slot on new master, err:", err)
			return false, err
		}
	}

	glog.V(4).Infof("new nodes status: \n %v", nodes)

	// Set the cluster status
	rCluster.Status = rapi.ClusterStatusOK
	// wait a bit for the cluster to propagate configuration to reduce warning logs because of temporary inconsistency
	time.Sleep(1 * time.Second)
	return asChanged, nil
}

func newSubmarineCluster(admin submarine.AdminInterface, cluster *rapi.SubmarineCluster) (*submarine.Cluster, submarine.Nodes, error) {
	infos, err := admin.GetClusterInfos()
	if submarine.IsPartialError(err) {
		glog.Errorf("Error getting consolidated view of the cluster err: %v", err)
		return nil, nil, err
	}

	// now we can trigger the rebalance
	nodes := infos.GetNodes()

	// build submarine cluster vision
	rCluster := &submarine.Cluster{
		Name:      cluster.Name,
		Namespace: cluster.Namespace,
		Nodes:     make(map[string]*submarine.Node),
	}

	for _, node := range nodes {
		rCluster.Nodes[node.ID] = node
	}

	for _, node := range cluster.Status.Cluster.Nodes {
		if rNode, ok := rCluster.Nodes[node.ID]; ok {
			rNode.Pod = node.Pod
		}
	}

	return rCluster, nodes, nil
}

// manageRollingUpdate used to manage properly a cluster rolling update if the podtemplate spec has changed
func (c *Controller) manageRollingUpdate(admin submarine.AdminInterface, cluster *rapi.SubmarineCluster, rCluster *submarine.Cluster, nodes submarine.Nodes) (bool, error) {
	return true, nil
}

// managePodScaleDown used to manage properly the scale down of a cluster
func (c *Controller) managePodScaleDown(admin submarine.AdminInterface, cluster *rapi.SubmarineCluster, rCluster *submarine.Cluster, nodes submarine.Nodes) (bool, error) {
	return true, nil
}
