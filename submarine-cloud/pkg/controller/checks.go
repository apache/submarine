package controller

import (
	"reflect"

	rapi "github.com/apache/submarine/submarine-cloud/pkg/apis/submarine/v1alpha1"
	podctrl "github.com/apache/submarine/submarine-cloud/pkg/controller/pod"
	"github.com/golang/glog"
	kapi "k8s.io/api/core/v1"
)

// Divide pods for lost and other
func filterLostNodes(pods []*kapi.Pod) (ok []*kapi.Pod, ko []*kapi.Pod) {
	for _, pod := range pods {
		if pod.Status.Reason == "NodeLost" {
			ko = append(ko, pod)
		} else {
			ok = append(ok, pod)
		}
	}
	return ok, ko
}

func compareStatus(old, new *rapi.SubmarineClusterClusterStatus) bool {
	if compareStringValue("ClusterStatus", string(old.Status), string(new.Status)) {
		return true
	}
	if compareInts("NbPods", old.NbPods, new.NbPods) {
		return true
	}
	if compareInts("NbPodsReady", old.NbPodsReady, new.NbPodsReady) {
		return true
	}
	if compareInts("NbSubmarineRunning", old.NbSubmarineRunning, new.NbSubmarineRunning) {
		return true
	}
	if compareInts("NumberOfMaster", old.NumberOfMaster, new.NumberOfMaster) {
		return true
	}
	if compareInts("MinReplicationFactor", old.MinReplicationFactor, new.MinReplicationFactor) {
		return true
	}
	if compareInts("MaxReplicationFactor", old.MaxReplicationFactor, new.MaxReplicationFactor) {
		return true
	}
	if compareStringValue("ClusterStatus", string(old.Status), string(new.Status)) {
		return true
	}
	if compareStringValue("NodesPlacement", string(old.NodesPlacement), string(new.NodesPlacement)) {
		return true
	}
	if compareInts("len(Nodes)", int32(len(old.Nodes)), int32(len(new.Nodes))) {
		return true
	}

	if len(old.Nodes) != len(new.Nodes) {
		return true
	}
	for _, nodeA := range old.Nodes {
		found := false
		for _, nodeB := range new.Nodes {
			if nodeA.ID == nodeB.ID {
				found = true
				if compareNodes(&nodeA, &nodeB) {
					return true
				}
			}
		}
		if !found {
			return true
		}
	}

	return false
}

func compareStringValue(name string, old, new string) bool {
	if old != new {
		glog.V(6).Infof("compare %s: %s - %s", name, old, new)
		return true
	}

	return false
}

func compareInts(name string, old, new int32) bool {
	if old != new {
		glog.Infof("compare status.%s: %d - %d", name, old, new)
		return true
	}

	return false
}

func compareNodes(nodeA, nodeB *rapi.SubmarineClusterNode) bool {
	if compareStringValue("Node.IP", nodeA.IP, nodeB.IP) {
		return true
	}
	if compareStringValue("Node.MasterRef", nodeA.MasterRef, nodeB.MasterRef) {
		return true
	}
	if compareStringValue("Node.PodName", nodeA.PodName, nodeB.PodName) {
		return true
	}
	if compareStringValue("Node.Port", nodeA.Port, nodeB.Port) {
		return true
	}
	if compareStringValue("Node.Role", string(nodeA.Role), string(nodeB.Role)) {
		return true
	}

	sizeSlotsA := 0
	sizeSlotsB := 0
	if nodeA.Slots != nil {
		sizeSlotsA = len(nodeA.Slots)
	}
	if nodeB.Slots != nil {
		sizeSlotsB = len(nodeB.Slots)
	}
	if sizeSlotsA != sizeSlotsB {
		glog.Infof("compare Node.Slote size: %d - %d", sizeSlotsA, sizeSlotsB)
		return true
	}

	if (sizeSlotsA != 0) && !reflect.DeepEqual(nodeA.Slots, nodeB.Slots) {
		glog.Infof("compare Node.Slote deepEqual: %v - %v", nodeA.Slots, nodeB.Slots)
		return true
	}

	return false
}

func needClusterOperation(cluster *rapi.SubmarineCluster) bool {
	/*
		if needRollingUpdate(cluster) {
			glog.V(6).Info("needClusterOperation---needRollingUpdate")
			return true
		}

		if needMorePods(cluster) {
			glog.V(6).Info("needClusterOperation---needMorePods")
			return true
		}

		if needLessPods(cluster) {
			glog.Info("needClusterOperation---needLessPods")
			return true
		}

		if compareIntValue("NumberOfMaster", &cluster.Status.Cluster.NumberOfMaster, cluster.Spec.NumberOfMaster) {
			glog.V(6).Info("needClusterOperation---NumberOfMaster")
			return true
		}

		if compareIntValue("MinReplicationFactor", &cluster.Status.Cluster.MinReplicationFactor, cluster.Spec.ReplicationFactor) {
			glog.V(6).Info("needClusterOperation---MinReplicationFactor")
			return true
		}

		if compareIntValue("MaxReplicationFactor", &cluster.Status.Cluster.MaxReplicationFactor, cluster.Spec.ReplicationFactor) {
			glog.V(6).Info("needClusterOperation---MaxReplicationFactor")
			return true
		}*/

	return false
}

func needMorePods(cluster *rapi.SubmarineCluster) bool {
	// Expected number of Pods depends on replication factor and Master number
	nbPodNeed := *cluster.Spec.NumberOfMaster * (1 + *cluster.Spec.ReplicationFactor)
	glog.Infof("nbPodNeed=%d, *cluster.Spec.NumberOfMaster=%d, *cluster.Spec.ReplicationFactor=%d", nbPodNeed, *cluster.Spec.NumberOfMaster, *cluster.Spec.ReplicationFactor)

	// If not all Pods are ready, do nothing
	glog.Infof("cluster.Status.Cluster.NbPods=%d", cluster.Status.Cluster.NbPods)
	glog.Infof("cluster.Status.Cluster.NbPodsReady=%d", cluster.Status.Cluster.NbPodsReady)
	if cluster.Status.Cluster.NbPods != cluster.Status.Cluster.NbPodsReady {
		return false
	}
	output := false
	if cluster.Status.Cluster.NbPods < nbPodNeed {
		glog.V(4).Infof("Not enough Pods running to apply the cluster [%s-%s] spec, current %d, needed %d ", cluster.Namespace, cluster.Name, cluster.Status.Cluster.NbPodsReady, nbPodNeed)
		output = true
	}

	return output
}

func needLessPods(cluster *rapi.SubmarineCluster) bool {
	nbPodNeed := *cluster.Spec.NumberOfMaster * (1 + *cluster.Spec.ReplicationFactor)

	if cluster.Status.Cluster.NbPods != cluster.Status.Cluster.NbPodsReady {
		return false
	}
	output := false
	if cluster.Status.Cluster.NbPods > nbPodNeed {
		glog.V(4).Infof("To many Pods running, needs to scale down the cluster [%s-%s], current %d, needed %d ", cluster.Namespace, cluster.Name, cluster.Status.Cluster.NbPods, nbPodNeed)
		output = true
	}
	return output
}

func needRollingUpdate(cluster *rapi.SubmarineCluster) bool {
	return !comparePodsWithPodTemplate(cluster)
}

func comparePodsWithPodTemplate(cluster *rapi.SubmarineCluster) bool {
	clusterPodSpecHash, _ := podctrl.GenerateMD5Spec(&cluster.Spec.PodTemplate.Spec)
	for _, node := range cluster.Status.Cluster.Nodes {
		if node.Pod == nil {
			continue
		}
		if !comparePodSpecMD5Hash(clusterPodSpecHash, node.Pod) {
			return false
		}
	}

	return true
}

func comparePodSpecMD5Hash(hash string, pod *kapi.Pod) bool {
	if val, ok := pod.Annotations[rapi.PodSpecMD5LabelKey]; ok {
		if val != hash {
			return false
		}
	} else {
		return false
	}

	return true
}
