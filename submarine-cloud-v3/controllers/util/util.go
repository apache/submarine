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

package util

import (
	submarineapacheorgv1alpha1 "github.com/apache/submarine/submarine-cloud-v3/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"reflect"
)

// GetSubmarineCommon will get `spec.common` and initialize it if it is nil to prevent NPE
func GetSubmarineCommon(submarine *submarineapacheorgv1alpha1.Submarine) *submarineapacheorgv1alpha1.SubmarineCommon {
	common := submarine.Spec.Common
	if common == nil {
		common = &submarineapacheorgv1alpha1.SubmarineCommon{}
	}
	return common
}

// GetSubmarineCommonImage will get `spec.common.image` and initialize it if it is nil to prevent NPE
func GetSubmarineCommonImage(submarine *submarineapacheorgv1alpha1.Submarine) *submarineapacheorgv1alpha1.CommonImage {
	common := GetSubmarineCommon(submarine)
	image := &common.Image
	if image == nil {
		image = &submarineapacheorgv1alpha1.CommonImage{}
	}
	return image
}

// CompareSlice will determine if two slices are equal
func CompareSlice(a, b []string) bool {
	// If one is nil, the other must also be nil.
	if (a == nil) != (b == nil) {
		return false
	}
	if len(a) != len(b) {
		return false
	}
	for key, value := range a {
		if value != b[key] {
			return false
		}
	}
	return true
}

// CompareMap will determine if two maps are equal
func CompareMap(a, b map[string]string) bool {
	// If one is nil, the other must also be nil.
	if (a == nil) != (b == nil) {
		return false
	}
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if w, ok := b[k]; !ok || v != w {
			return false
		}
	}
	return true
}

// ComparePullSecrets will determine if two LocalObjectReferences are equal
func ComparePullSecrets(a, b []corev1.LocalObjectReference) bool {
	// If one is nil, the other must also be nil.
	if (a == nil) != (b == nil) {
		return false
	}
	if len(a) != len(b) {
		return false
	}
	for key, value := range a {
		if value.Name != b[key].Name {
			return false
		}
	}
	return true
}

// CompareEnv will determine if two EnvVars are equal
func CompareEnv(a, b []corev1.EnvVar) bool {
	// If one is nil, the other must also be nil.
	if (a == nil) != (b == nil) {
		return false
	}
	if len(a) != len(b) {
		return false
	}
	for key, value := range a {
		cv := b[key]
		if value.Name != cv.Name || value.Value != cv.Value || !reflect.DeepEqual(value.ValueFrom, cv.ValueFrom) {
			return false
		}
	}
	return true
}

// GetSecretData will return secret data ( map[string]string )
func GetSecretData(secret *corev1.Secret) map[string]string {
	if secret.StringData != nil {
		return secret.StringData
	} else {
		var parsedData map[string]string
		parsedData = make(map[string]string)
		for key, value := range secret.Data {
			parsedData[key] = string(value)
		}
		return parsedData
	}
}

// CompareSecret will determine if two Secrets are equal
func CompareSecret(oldSecret, newSecret *corev1.Secret) bool {
	return CompareMap(GetSecretData(oldSecret), GetSecretData(newSecret))
}
