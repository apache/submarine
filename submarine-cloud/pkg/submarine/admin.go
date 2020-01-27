package submarine

import (
	"github.com/golang/glog"
	"time"
)

// AdminInterface submarine cluster admin interface
type AdminInterface interface {
	// Connections returns the connection map of all clients
	Connections() AdminConnectionsInterface
	// Close the admin connections
	Close()
	// InitSubmarineCluster used to configure the first node of a cluster
	InitSubmarineCluster(addr string) error
	// GetClusterInfos get node infos for all nodes
	GetClusterInfos() (*ClusterInfos, error)
	// GetClusterInfosSelected return the Nodes infos for all nodes selected in the cluster
	//GetClusterInfosSelected(addrs []string) (*ClusterInfos, error)
	// AttachNodeToCluster command use to connect a Node to the cluster
	// the connection will be done on a random node part of the connection pool
	AttachNodeToCluster(addr string) error
	// AttachSlaveToMaster attach a slave to a master node
	AttachSlaveToMaster(slave *Node, master *Node) error
	// DetachSlave dettach a slave to its master
	//DetachSlave(slave *Node) error
	// StartFailover execute the failover of the Submarine Master corresponding to the addr
	StartFailover(addr string) error
	// ForgetNode execute the Submarine command to force the cluster to forgot the the Node
	ForgetNode(id string) error
	// ForgetNodeByAddr execute the Submarine command to force the cluster to forgot the the Node
	ForgetNodeByAddr(id string) error
	// SetSlots exect the submarine command to set slots in a pipeline, provide
	// and empty nodeID if the set slots commands doesn't take a nodeID in parameter
	//SetSlots(addr string, action string, slots []Slot, nodeID string) error
	// AddSlots exect the submarine command to add slots in a pipeline
	//AddSlots(addr string, slots []Slot) error
	// DelSlots exec the submarine command to del slots in a pipeline
	//DelSlots(addr string, slots []Slot) error
	// GetKeysInSlot exec the submarine command to get the keys in the given slot on the node we are connected to
	//GetKeysInSlot(addr string, slot Slot, batch int, limit bool) ([]string, error)
	// CountKeysInSlot exec the submarine command to count the keys given slot on the node
	//CountKeysInSlot(addr string, slot Slot) (int64, error)
	// MigrateKeys from addr to destination node. returns number of slot migrated. If replace is true, replace key on busy error
	//MigrateKeys(addr string, dest *Node, slots []Slot, batch, timeout int, replace bool) (int, error)
	// FlushAndReset reset the cluster configuration of the node, the node is flushed in the same pipe to ensure reset works
	FlushAndReset(addr string, mode string) error
	// FlushAll flush all keys in cluster
	FlushAll()
	// GetHashMaxSlot get the max slot value
	//GetHashMaxSlot() Slot
	//RebuildConnectionMap rebuild the connection map according to the given addresses
	//RebuildConnectionMap(addrs []string, options *AdminOptions)
}

// AdminOptions optional options for submarine admin
type AdminOptions struct {
	ConnectionTimeout  time.Duration
	ClientName         string
	RenameCommandsFile string
}

// Admin wraps submarine cluster admin logic
type Admin struct {
	///hashMaxSlots Slot
	cnx AdminConnectionsInterface
}

func (a Admin) Connections() AdminConnectionsInterface {
	return a.cnx
}

func (a Admin) Close() {
	a.Connections().Reset()
}

func (a Admin) InitSubmarineCluster(addr string) error {
	panic("implement me")
}

func (a Admin) GetClusterInfos() (*ClusterInfos, error) {
	glog.V(1).Info("GetClusterInfos")

	infos := NewClusterInfos()
	clusterErr := NewClusterInfosError()

	for addr, c := range a.Connections().GetAll() {
		nodeinfos, err := a.getInfos(c, addr)
		if err != nil {
			infos.Status = ClusterInfosPartial
			clusterErr.partial = true
			clusterErr.errs[addr] = err
			continue
		}
		if nodeinfos.Node != nil && nodeinfos.Node.IPPort() == addr {
			infos.Infos[addr] = nodeinfos
		} else {
			glog.Warningf("Bad node info retreived from %s", addr)
		}
	}

	if len(clusterErr.errs) == 0 {
		clusterErr.inconsistent = !infos.ComputeStatus()
	}
	if infos.Status == ClusterInfosConsistent {
		return infos, nil
	}
	return infos, clusterErr
}

func (a Admin) AttachNodeToCluster(addr string) error {
	panic("implement me")
}

func (a Admin) AttachSlaveToMaster(slave *Node, master *Node) error {
	panic("implement me")
}

func (a Admin) StartFailover(addr string) error {
	panic("implement me")
}

func (a Admin) ForgetNode(id string) error {
	panic("implement me")
}

func (a Admin) ForgetNodeByAddr(id string) error {
	panic("implement me")
}

func (a Admin) FlushAndReset(addr string, mode string) error {
	panic("implement me")
}

func (a Admin) FlushAll() {
	panic("implement me")
}

// NewAdmin returns new AdminInterface instance
// at the same time it connects to all Submarine Nodes thanks to the addrs list
func NewAdmin(addrs []string, options *AdminOptions) AdminInterface {
	a := &Admin{
		//hashMaxSlots: defaultHashMaxSlots,
	}

	// perform initial connections
	a.cnx = NewAdminConnections(addrs, options)

	return a
}

func (a *Admin) getInfos(c ClientInterface, addr string) (*NodeInfos, error) {
	/*
		resp := c.Cmd("CLUSTER", "NODES")
		if err := a.Connections().ValidateResp(resp, addr, "Unable to retrieve Node Info"); err != nil {
			return nil, err
		}

		var raw string
		var err error
		raw, err = resp.Str()

		if err != nil {
			return nil, fmt.Errorf("Wrong format from CLUSTER NODES: %v", err)
		}
	*/
	var raw string = ""
	nodeInfos := DecodeNodeInfos(&raw, addr)

	/*
		if glog.V(3) {
			//Retrieve server info for debugging
			resp = c.Cmd("INFO", "SERVER")
			if err = a.Connections().ValidateResp(resp, addr, "Unable to retrieve Node Info"); err != nil {
				return nil, err
			}
			raw, err = resp.Str()
			if err != nil {
				return nil, fmt.Errorf("Wrong format from INFO SERVER: %v", err)
			}

			var serverStartTime time.Time
			serverStartTime, err = DecodeNodeStartTime(&raw)

			if err != nil {
				return nil, err
			}

			nodeInfos.Node.ServerStartTime = serverStartTime
		}*/

	return nodeInfos, nil
}
