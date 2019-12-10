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

package pod

import (
	rapi "github.com/apache/submarine/submarine-cloud/pkg/apis/submarine/v1alpha1"
	"github.com/golang/glog"
	kapiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/record"
)

// SubmarineClusterControlInteface interface for the SubmarineClusterPodControl
type SubmarineClusterControlInteface interface {
	// GetSubmarineClusterPods return list of Pod attached to a SubmarineCluster
	GetSubmarineClusterPods(submarineCluster *rapi.SubmarineCluster) ([]*kapiv1.Pod, error)
	// CreatePod used to create a Pod from the SubmarineCluster pod template
	CreatePod(submarineCluster *rapi.SubmarineCluster) (*kapiv1.Pod, error)
	// DeletePod used to delete a pod from its name
	DeletePod(submarineCluster *rapi.SubmarineCluster, podName string) error
	// DeletePodNow used to delete now (force) a pod from its name
	DeletePodNow(submarineCluster *rapi.SubmarineCluster, podName string) error
}

// SubmarineClusterControl contains requieres accessor to managing the SubmarineCluster pods
type SubmarineClusterControl struct {
	PodLister  corev1listers.PodLister
	KubeClient clientset.Interface
	Recorder   record.EventRecorder
}

// NewSubmarineClusterControl builds and returns new NewSubmarineClusterControl instance
func NewSubmarineClusterControl(lister corev1listers.PodLister, client clientset.Interface, rec record.EventRecorder) *SubmarineClusterControl {
	glog.Infof("NewSubmarineClusterControl()")
	ctrl := &SubmarineClusterControl{
		PodLister:  lister,
		KubeClient: client,
		Recorder:   rec,
	}
	return ctrl
}

// GetSubmarineClusterPods return list of Pod attached to a SubmarineCluster
func (p *SubmarineClusterControl) GetSubmarineClusterPods(submarineCluster *rapi.SubmarineCluster) ([]*kapiv1.Pod, error) {
	glog.Infof("GetSubmarineClusterPods()")
	return nil, nil
}

// CreatePod used to create a Pod from the SubmarineCluster pod template
func (p *SubmarineClusterControl) CreatePod(submarineCluster *rapi.SubmarineCluster) (*kapiv1.Pod, error) {
	glog.Infof("CreatePod()")
	return nil, nil
}

// DeletePod used to delete a pod from its name
func (p *SubmarineClusterControl) DeletePod(submarineCluster *rapi.SubmarineCluster, podName string) error {
	glog.V(6).Infof("DeletePod: %s/%s", submarineCluster.Namespace, podName)
	return p.deletePodGracefullperiode(submarineCluster, podName, nil)
}

// DeletePodNow used to delete now (force) a pod from its name
func (p *SubmarineClusterControl) DeletePodNow(submarineCluster *rapi.SubmarineCluster, podName string) error {
	glog.V(6).Infof("DeletePod: %s/%s", submarineCluster.Namespace, podName)
	now := int64(0)
	return p.deletePodGracefullperiode(submarineCluster, podName, &now)
}

// DeletePodNow used to delete now (force) a pod from its name
func (p *SubmarineClusterControl) deletePodGracefullperiode(submarineCluster *rapi.SubmarineCluster, podName string, period *int64) error {
	glog.Infof("deletePodGracefullperiode()")
	return p.KubeClient.CoreV1().Pods(submarineCluster.Namespace).Delete(podName, &metav1.DeleteOptions{GracePeriodSeconds: period})
}
