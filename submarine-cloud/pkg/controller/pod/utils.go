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
	"fmt"
	"k8s.io/apimachinery/pkg/labels"

	sapi "github.com/apache/submarine/submarine-cloud/pkg/apis/submarine/v1alpha1"
)

// GetLabelsSet return labels associated to the submarine-node pods
func GetLabelsSet(submarineCluster *sapi.SubmarineCluster) (labels.Set, error) {
	desiredLabels := labels.Set{}
	if submarineCluster == nil {
		return desiredLabels, fmt.Errorf("submarineCluster nil pointer")
	}
	if submarineCluster.Spec.AdditionalLabels != nil {
		desiredLabels = submarineCluster.Spec.AdditionalLabels
	}
	if submarineCluster.Spec.PodTemplate != nil {
		for k, v := range submarineCluster.Spec.PodTemplate.Labels {
			desiredLabels[k] = v
		}
	}
	desiredLabels[sapi.ClusterNameLabelKey] = submarineCluster.Name // add submarineCluster name to the Pod labels
	return desiredLabels, nil
}

// CreateSubmarineClusterLabelSelector creates label selector to select the jobs related to a submarineCluster, stepName
func CreateSubmarineClusterLabelSelector(submarineCluster *sapi.SubmarineCluster) (labels.Selector, error) {
	set, err := GetLabelsSet(submarineCluster)
	if err != nil {
		return nil, err
	}
	return labels.SelectorFromSet(set), nil
}

// GetAnnotationsSet return a labels.Set of annotation from the SubmarineCluster
func GetAnnotationsSet(submarineCluster *sapi.SubmarineCluster) (labels.Set, error) {
	desiredAnnotations := make(labels.Set)
	for k, v := range submarineCluster.Annotations {
		desiredAnnotations[k] = v
	}

	// TODO: add createdByRef
	return desiredAnnotations, nil // no error for the moment, when we'll add createdByRef an error could be returned
}
