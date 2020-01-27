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
