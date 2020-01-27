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
