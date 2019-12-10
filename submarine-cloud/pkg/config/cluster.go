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

import "github.com/spf13/pflag"

// Cluster used to store all Submarine Cluster configuration information
type Cluster struct {
	Namespace   string
	NodeService string
}

// AddFlags use to add the Submarine-Cluster Config flags to the command line
func (c *Cluster) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&c.Namespace, "ns", "", "Submarine-node k8s namespace")
	fs.StringVar(&c.NodeService, "rs", "", "Submarine-node k8s service name")

}
