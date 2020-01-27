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
package operator

import (
	"github.com/apache/submarine/submarine-cloud/pkg/config"
	"github.com/spf13/pflag"
)

// Config contains configuration for submarine-operator
type Config struct {
	KubeConfigFile string
	Master         string
	ListenAddr     string
	Submarine      config.Submarine
}

// NewSubmarineOperatorConfig builds and returns a submarine-operator Config
func NewSubmarineOperatorConfig() *Config {
	return &Config{}
}

// AddFlags add cobra flags to populate Config
func (c *Config) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&c.KubeConfigFile, "kubeconfig", c.KubeConfigFile, "Location of kubecfg file for access to kubernetes master service")
	fs.StringVar(&c.Master, "master", c.Master, "The address of the Kubernetes API server. Overrides any value in kubeconfig. Only required if out-of-cluster.")
	fs.StringVar(&c.ListenAddr, "addr", "0.0.0.0:8080", "listen address of the http server which serves kubernetes probes and prometheus endpoints")
	c.Submarine.AddFlags(fs)
}
