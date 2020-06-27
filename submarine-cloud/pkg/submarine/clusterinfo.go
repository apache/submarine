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

const (
	// ClusterInfosUnset status of the cluster info: no data set
	ClusterInfosUnset = "Unset"
	// ClusterInfosPartial status of the cluster info: data is not complete (some nodes didn't respond)
	ClusterInfosPartial = "Partial"
	// ClusterInfosInconsistent status of the cluster info: nodesinfos is not consistent between nodes
	ClusterInfosInconsistent = "Inconsistent"
	// ClusterInfosConsistent status of the cluster info: nodeinfos is complete and consistent between nodes
	ClusterInfosConsistent = "Consistent"
)

// ClusterInfos represents the node infos for all nodes of the cluster
type ClusterInfos struct {
	Infos  map[string]*NodeInfos
	Status string
}

// NodeInfos representation of a node info, i.e. data returned by the CLUSTER NODE submarine command
// Node is the information of the targetted node
// Friends are the view of the other nodes from the targetted node
type NodeInfos struct {
	Node    *Node
	Friends Nodes
}

// GetNodes returns a nodeSlice view of the cluster
// the slice if formed from how each node see itself
// you should check the Status before doing it, to wait for a consistent view
func (c *ClusterInfos) GetNodes() Nodes {
	nodes := Nodes{}
	for _, nodeinfos := range c.Infos {
		nodes = append(nodes, nodeinfos.Node)
	}
	return nodes
}

// NewNodeInfos returns an instance of NodeInfo
func NewNodeInfos() *NodeInfos {
	return &NodeInfos{
		Node:    NewDefaultNode(),
		Friends: Nodes{},
	}
}

// NewClusterInfos returns an instance of ClusterInfos
func NewClusterInfos() *ClusterInfos {
	return &ClusterInfos{
		Infos:  make(map[string]*NodeInfos),
		Status: ClusterInfosUnset,
	}
}

// DecodeNodeInfos decode from the cmd output the Submarine nodes info. Second argument is the node on which we are connected to request info
func DecodeNodeInfos(input *string, addr string) *NodeInfos {
	infos := NewNodeInfos()
	/*
		lines := strings.Split(*input, "\n")
		for _, line := range lines {
			values := strings.Split(line, " ")
			if len(values) < 8 {
				// last line is always empty
				glog.V(7).Infof("Not enough values in line split, ignoring line: '%s'", line)
				continue
			} else {
				node := NewDefaultNode()

				node.ID = values[0]
				//remove trailing port for cluster internal protocol
				ipPort := strings.Split(values[1], "@")
				if ip, port, err := net.SplitHostPort(ipPort[0]); err == nil {
					node.IP = ip
					node.Port = port
					if ip == "" {
						// ip of the node we are connecting to is sometime empty
						node.IP, _, _ = net.SplitHostPort(addr)
					}
				} else {
					glog.Errorf("Error while decoding node info for node '%s', cannot split ip:port ('%s'): %v", node.ID, values[1], err)
				}
				node.SetRole(values[2])
				node.SetFailureStatus(values[2])
				node.SetReferentMaster(values[3])
				if i, err := strconv.ParseInt(values[4], 10, 64); err == nil {
					node.PingSent = i
				}
				if i, err := strconv.ParseInt(values[5], 10, 64); err == nil {
					node.PongRecv = i
				}
				if i, err := strconv.ParseInt(values[6], 10, 64); err == nil {
					node.ConfigEpoch = i
				}
				node.SetLinkStatus(values[7])

				for _, slot := range values[8:] {
					if s, importing, migrating, err := DecodeSlotRange(slot); err == nil {
						node.Slots = append(node.Slots, s...)
						if importing != nil {
							node.ImportingSlots[importing.SlotID] = importing.FromNodeID
						}
						if migrating != nil {
							node.MigratingSlots[migrating.SlotID] = migrating.ToNodeID
						}
					}
				}

				if strings.HasPrefix(values[2], "myself") {
					infos.Node = node
					glog.V(7).Infof("Getting node info for node: '%s'", node)
				} else {
					infos.Friends = append(infos.Friends, node)
					glog.V(7).Infof("Adding node to slice: '%s'", node)
				}
			}
		}*/

	return infos
}

// ComputeStatus check the ClusterInfos status based on the current data
// the status ClusterInfosPartial is set while building the clusterinfos
// if already set, do nothing
// returns true if consistent or if another error
func (c *ClusterInfos) ComputeStatus() bool {
	if c.Status != ClusterInfosUnset {
		return false
	}
	return true

	/*
		consistencyStatus := false

		consolidatedView := c.GetNodes().SortByFunc(LessByID)
		consolidatedSignature := getConfigSignature(consolidatedView)
		glog.V(7).Infof("Consolidated view:\n%s", consolidatedSignature)
		for addr, nodeinfos := range c.Infos {
			nodesView := append(nodeinfos.Friends, nodeinfos.Node).SortByFunc(LessByID)
			nodeSignature := getConfigSignature(nodesView)
			glog.V(7).Infof("Node view from %s (ID: %s):\n%s", addr, nodeinfos.Node.ID, nodeSignature)
			if !reflect.DeepEqual(consolidatedSignature, nodeSignature) {
				glog.V(4).Info("Temporary inconsistency between nodes is possible. If the following inconsistency message persists for more than 20 mins, any cluster operation (scale, rolling update) should be avoided before the message is gone")
				glog.V(4).Infof("Inconsistency from %s: \n%s\nVS\n%s", addr, consolidatedSignature, nodeSignature)
				c.Status = ClusterInfosInconsistent
			}
		}
		if c.Status == ClusterInfosUnset {
			c.Status = ClusterInfosConsistent
			consistencyStatus = true
		}
		return consistencyStatus*/
}
