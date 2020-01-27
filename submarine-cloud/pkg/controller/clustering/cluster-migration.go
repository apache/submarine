package clustering

import (
	"fmt"
	v1 "github.com/apache/submarine/submarine-cloud/pkg/apis/submarine/v1alpha1"
	"github.com/apache/submarine/submarine-cloud/pkg/submarine"
	"github.com/golang/glog"
)

// DispatchMasters used to select nodes with master roles
func DispatchMasters(cluster *submarine.Cluster, nodes submarine.Nodes, nbMaster int32, admin submarine.AdminInterface) (submarine.Nodes, submarine.Nodes, submarine.Nodes, error) {
	glog.Info("Start dispatching slots to masters nb nodes: ", len(nodes))
	var allMasterNodes submarine.Nodes
	// First loop get Master with already Slots assign on it
	currentMasterNodes := nodes.FilterByFunc(submarine.IsMasterWithSlot)
	allMasterNodes = append(allMasterNodes, currentMasterNodes...)

	// add also available Master without slot
	currentMasterWithNoSlot := nodes.FilterByFunc(submarine.IsMasterWithNoSlot)
	allMasterNodes = append(allMasterNodes, currentMasterWithNoSlot...)
	glog.V(2).Info("Master with No slot:", len(currentMasterWithNoSlot))

	newMasterNodesSmartSelection, besteffort, err := PlaceMasters(cluster, currentMasterNodes, currentMasterWithNoSlot, nbMaster)

	glog.V(2).Infof("Total masters: %d - target %d - selected: %d", len(allMasterNodes), nbMaster, len(newMasterNodesSmartSelection))
	if err != nil {
		return submarine.Nodes{}, submarine.Nodes{}, submarine.Nodes{}, fmt.Errorf("Not Enough Master available current:%d target:%d, err:%v", len(allMasterNodes), nbMaster, err)
	}

	newMasterNodesSmartSelection = newMasterNodesSmartSelection.SortByFunc(func(a, b *submarine.Node) bool { return a.ID < b.ID })

	cluster.Status = v1.ClusterStatusCalculatingRebalancing
	if besteffort {
		cluster.NodesPlacement = v1.NodesPlacementInfoBestEffort
	} else {
		cluster.NodesPlacement = v1.NodesPlacementInfoOptimal
	}

	return newMasterNodesSmartSelection, currentMasterNodes, allMasterNodes, nil
}

// DispatchSlotToNewMasters used to dispatch Slot to the new master nodes
func DispatchSlotToNewMasters(cluster *submarine.Cluster, admin submarine.AdminInterface, newMasterNodes, currentMasterNodes, allMasterNodes submarine.Nodes) error {
	return nil
}
