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

package config

import (
	"fmt"
	"path"

	"github.com/spf13/pflag"
)

const (
	// DefaultSubmarineTimeout default Submarine timeout (ms)
	DefaultSubmarineTimeout = 2000
	//DefaultClusterNodeTimeout default cluster node timeout (ms)
	//The maximum amount of time a Submarine Cluster node can be unavailable, without it being considered as failing
	DefaultClusterNodeTimeout = 2000
	// SubmarineRenameCommandsDefaultPath default path to volume storing rename commands
	SubmarineRenameCommandsDefaultPath = "/etc/secret-volume"
	// SubmarineRenameCommandsDefaultFile default file name containing rename commands
	SubmarineRenameCommandsDefaultFile = ""
	// SubmarineConfigFileDefault default config file path
	SubmarineConfigFileDefault = "/submarine-conf/submarine.conf"
	// SubmarineServerBinDefault default binary name
	SubmarineServerBinDefault = "submarine-server"
	// SubmarineServerPortDefault default Submarine port
	SubmarineServerPortDefault = "6379"
	// SubmarineMaxMemoryDefault default Submarine max memory
	SubmarineMaxMemoryDefault = 0
	// SubmarineMaxMemoryPolicyDefault default Submarine max memory evition policy
	SubmarineMaxMemoryPolicyDefault = "noeviction"
)

// Submarine used to store all Submarine configuration information
type Submarine struct {
	DialTimeout        int
	ClusterNodeTimeout int
	ConfigFileName     string
	renameCommandsPath string
	renameCommandsFile string
	HTTPServerAddr     string
	ServerBin          string
	ServerPort         string
	ServerIP           string
	MaxMemory          uint32
	MaxMemoryPolicy    string
	ConfigFiles        []string
}

// AddFlags use to add the Submarine Config flags to the command line
func (r *Submarine) AddFlags(fs *pflag.FlagSet) {
	fs.IntVar(&r.DialTimeout, "rdt", DefaultSubmarineTimeout, "Submarine dial timeout (ms)")
	fs.IntVar(&r.ClusterNodeTimeout, "cluster-node-timeout", DefaultClusterNodeTimeout, "Submarine node timeout (ms)")
	fs.StringVar(&r.ConfigFileName, "c", SubmarineConfigFileDefault, "Submarine config file path")
	fs.StringVar(&r.renameCommandsPath, "rename-command-path", SubmarineRenameCommandsDefaultPath, "Path to the folder where rename-commands option for Submarine are available")
	fs.StringVar(&r.renameCommandsFile, "rename-command-file", SubmarineRenameCommandsDefaultFile, "Name of the file where rename-commands option for Submarine are available, disabled if empty")
	fs.Uint32Var(&r.MaxMemory, "max-memory", SubmarineMaxMemoryDefault, "Submarine max memory")
	fs.StringVar(&r.MaxMemoryPolicy, "max-memory-policy", SubmarineMaxMemoryPolicyDefault, "Submarine max memory evition policy")
	fs.StringVar(&r.ServerBin, "bin", SubmarineServerBinDefault, "Submarine server binary file name")
	fs.StringVar(&r.ServerPort, "port", SubmarineServerPortDefault, "Submarine server listen port")
	fs.StringVar(&r.ServerIP, "ip", "", "Submarine server listen ip")
	fs.StringArrayVar(&r.ConfigFiles, "config-file", []string{}, "Location of Submarine configuration file that will be include in the ")

}

// GetRenameCommandsFile return the path to the rename command file, or empty string if not define
func (r *Submarine) GetRenameCommandsFile() string {
	if r.renameCommandsFile == "" {
		return ""
	}
	return path.Join(r.renameCommandsPath, r.renameCommandsFile)
}

// String stringer interface
func (r Submarine) String() string {
	var output string
	output += fmt.Sprintln("[ Submarine Configuration ]")
	output += fmt.Sprintln("- DialTimeout:", r.DialTimeout)
	output += fmt.Sprintln("- ClusterNodeTimeout:", r.ClusterNodeTimeout)
	output += fmt.Sprintln("- Rename commands:", r.GetRenameCommandsFile())
	output += fmt.Sprintln("- max-memory:", r.MaxMemory)
	output += fmt.Sprintln("- max-memory-policy:", r.MaxMemoryPolicy)
	output += fmt.Sprintln("- server-bin:", r.ServerBin)
	output += fmt.Sprintln("- server-port:", r.ServerPort)
	return output
}
