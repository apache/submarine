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

package controller

import (
	"github.com/apache/submarine/submarine-cloud-v2/pkg/helm"

	"k8s.io/klog/v2"
)

// subcharts: https://github.com/apache/submarine/tree/master/helm-charts/submarine/charts

func (c *Controller) installSubCharts(namespace string) error {
	// Install traefik
	// Reference: https://github.com/apache/submarine/tree/master/helm-charts/submarine/charts/traefik

	if !helm.CheckRelease("traefik", namespace) {
		klog.Info("[Helm] Install Traefik")
		c.charts = append(c.charts, helm.HelmInstallLocalChart(
			"traefik",
			"charts/traefik",
			"traefik",
			namespace,
			map[string]string{},
		))
	}

	if !helm.CheckRelease("notebook-controller", namespace) {
		klog.Info("[Helm] Install Notebook-Controller")
		c.charts = append(c.charts, helm.HelmInstallLocalChart(
			"notebook-controller",
			"charts/notebook-controller",
			"notebook-controller",
			namespace,
			map[string]string{},
		))
	}

	if !helm.CheckRelease("tfjob", namespace) {
		klog.Info("[Helm] Install TFjob")
		c.charts = append(c.charts, helm.HelmInstallLocalChart(
			"tfjob",
			"charts/tfjob",
			"tfjob",
			namespace,
			map[string]string{},
		))
	}

	if !helm.CheckRelease("pytorchjob", namespace) {
		klog.Info("[Helm] Install pytorchjob")
		c.charts = append(c.charts, helm.HelmInstallLocalChart(
			"pytorchjob",
			"charts/pytorchjob",
			"pytorchjob",
			namespace,
			map[string]string{},
		))
	}

	// TODO: maintain "error"
	// TODO: (sample-controller) controller.go:287 ~ 293

	return nil
}
