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
	"bytes"
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"fmt"
	rapi "github.com/apache/submarine/submarine-cloud/pkg/apis/submarine/v1alpha1"
	"github.com/golang/glog"
	"io"
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
	selector, err := CreateSubmarineClusterLabelSelector(submarineCluster)
	if err != nil {
		return nil, err
	}
	return p.PodLister.Pods(submarineCluster.Namespace).List(selector)
}

// CreatePod used to create a Pod from the SubmarineCluster pod template
func (p *SubmarineClusterControl) CreatePod(submarineCluster *rapi.SubmarineCluster) (*kapiv1.Pod, error) {
	glog.Infof("CreatePod()")
	pod, err := initPod(submarineCluster)
	if err != nil {
		return pod, err
	}
	glog.V(6).Infof("CreatePod: %s/%s", submarineCluster.Namespace, pod.Name)
	return p.KubeClient.CoreV1().Pods(submarineCluster.Namespace).Create(pod)
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

// GenerateMD5Spec used to generate the PodSpec MD5 hash
func GenerateMD5Spec(spec *kapiv1.PodSpec) (string, error) {
	b, err := json.Marshal(spec)
	if err != nil {
		return "", err
	}
	hash := md5.New()
	io.Copy(hash, bytes.NewReader(b))
	return hex.EncodeToString(hash.Sum(nil)), nil
}

// Add the necessary tags to the Pod. These tags are used to determine whether a Pod is managed by the Operator and associated with a SubmarineCluster
func initPod(submarineCluster *rapi.SubmarineCluster) (*kapiv1.Pod, error) {
	if submarineCluster == nil {
		return nil, fmt.Errorf("submarinecluster nil pointer")
	}

	desiredLabels, err := GetLabelsSet(submarineCluster)
	if err != nil {
		return nil, err
	}
	desiredAnnotations, err := GetAnnotationsSet(submarineCluster)
	if err != nil {
		return nil, err
	}
	PodName := fmt.Sprintf("submarinecluster-%s-", submarineCluster.Name)
	pod := &kapiv1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:       submarineCluster.Namespace,
			Labels:          desiredLabels,
			Annotations:     desiredAnnotations,
			GenerateName:    PodName,
			OwnerReferences: []metav1.OwnerReference{BuildOwnerReference(submarineCluster)},
		},
	}

	if submarineCluster.Spec.PodTemplate == nil {
		return nil, fmt.Errorf("submarinecluster[%s/%s] PodTemplate missing", submarineCluster.Namespace, submarineCluster.Name)
	}
	pod.Spec = *submarineCluster.Spec.PodTemplate.Spec.DeepCopy()

	// Generate a MD5 representing the PodSpec send
	hash, err := GenerateMD5Spec(&pod.Spec)
	if err != nil {
		return nil, err
	}
	pod.Annotations[rapi.PodSpecMD5LabelKey] = hash

	return pod, nil
}

// BuildOwnerReference used to build the OwnerReference from a SubmarineCluster
func BuildOwnerReference(cluster *rapi.SubmarineCluster) metav1.OwnerReference {
	controllerRef := metav1.OwnerReference{
		APIVersion: rapi.SchemeGroupVersion.String(),
		Kind:       rapi.ResourceKind,
		Name:       cluster.Name,
		UID:        cluster.UID,
		Controller: boolPtr(true),
	}

	return controllerRef
}

func boolPtr(value bool) *bool {
	return &value
}
