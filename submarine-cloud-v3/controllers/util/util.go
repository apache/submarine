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
