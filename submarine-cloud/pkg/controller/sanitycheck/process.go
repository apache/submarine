package sanitycheck

import (
	rapi "github.com/apache/submarine/submarine-cloud/pkg/apis/submarine/v1alpha1"
	"github.com/apache/submarine/submarine-cloud/pkg/config"
	"github.com/apache/submarine/submarine-cloud/pkg/controller/pod"
	"github.com/apache/submarine/submarine-cloud/pkg/submarine"
)

// RunSanityChecks function used to run all the sanity check on the current cluster
// Return actionDone = true if a modification has been made on the cluster
func RunSanityChecks(admin submarine.AdminInterface, config *config.Submarine, podControl pod.SubmarineClusterControlInteface, cluster *rapi.SubmarineCluster, infos *submarine.ClusterInfos, dryRun bool) (actionDone bool, err error) {
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
