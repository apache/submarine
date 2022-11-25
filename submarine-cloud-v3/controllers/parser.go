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

package controllers

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/pkg/errors"
	istiov1alpha3 "istio.io/client-go/pkg/apis/networking/v1alpha3"
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/yaml"
)

// PathToOSFile turn the file at the relativePath into a type of *os.File.
func pathToOSFile(relativePath string) (*os.File, error) {
	path, err := filepath.Abs(relativePath)
	if err != nil {
		return nil, errors.Wrap(err, fmt.Sprintf("failed generate absolute file path of %s", relativePath))
	}

	manifest, err := os.Open(path)
	if err != nil {
		return nil, errors.Wrap(err, fmt.Sprintf("failed to open file %s", path))
	}

	return manifest, nil
}

// ParseYaml
func parseYaml(relativePath, kind string) ([]byte, error) {
	var manifest *os.File
	var err error

	var marshaled []byte
	if manifest, err = pathToOSFile(relativePath); err != nil {
		return nil, err
	}

	decoder := yaml.NewYAMLOrJSONDecoder(manifest, 100)
	for {
		var out unstructured.Unstructured
		err = decoder.Decode(&out)
		if err != nil {
			// this would indicate it's malformed YAML.
			break
		}

		if out.GetKind() == kind {
			marshaled, err = out.MarshalJSON()
			break
		}
	}

	if err != io.EOF && err != nil {
		return nil, err
	}
	return marshaled, nil
}

// ParseServiceAccountYaml parse ServiceAccount from yaml file.
func ParseServiceAccountYaml(relativePath string) (*v1.ServiceAccount, error) {
	var serviceAccount v1.ServiceAccount
	marshaled, err := parseYaml(relativePath, "ServiceAccount")
	if err != nil {
		return nil, err
	}
	json.Unmarshal(marshaled, &serviceAccount)
	return &serviceAccount, nil
}

// ParseDeploymentYaml parse Deployment from yaml file.
func ParseDeploymentYaml(relativePath string) (*appsv1.Deployment, error) {
	var deployment appsv1.Deployment
	marshaled, err := parseYaml(relativePath, "Deployment")
	if err != nil {
		return nil, err
	}
	json.Unmarshal(marshaled, &deployment)
	return &deployment, nil
}

// ParseStatefulSetYaml parse StatefulSets from yaml file.
func ParseStatefulSetYaml(relativePath string) (*appsv1.StatefulSet, error) {
	var statefulset appsv1.StatefulSet
	marshaled, err := parseYaml(relativePath, "StatefulSet")
	if err != nil {
		return nil, err
	}
	json.Unmarshal(marshaled, &statefulset)
	return &statefulset, nil
}

// ParseServiceYaml parse Service from yaml file.
func ParseServiceYaml(relativePath string) (*v1.Service, error) {
	var service v1.Service
	marshaled, err := parseYaml(relativePath, "Service")
	if err != nil {
		return nil, err
	}
	json.Unmarshal(marshaled, &service)
	return &service, nil
}

// ParseRoleBindingYaml parse RoleBinding from yaml file.
func ParseRoleBindingYaml(relativePath string) (*rbacv1.RoleBinding, error) {
	var rolebinding rbacv1.RoleBinding
	marshaled, err := parseYaml(relativePath, "RoleBinding")
	if err != nil {
		return nil, err
	}
	json.Unmarshal(marshaled, &rolebinding)
	return &rolebinding, nil
}

// ParseRoleYaml parse Role from yaml file.
func ParseRoleYaml(relativePath string) (*rbacv1.Role, error) {
	var role rbacv1.Role
	marshaled, err := parseYaml(relativePath, "Role")
	if err != nil {
		return nil, err
	}
	json.Unmarshal(marshaled, &role)
	return &role, nil
}

// ParsePersistentVolumeClaimYaml parse PersistentVolumeClaimYaml from yaml file.
func ParsePersistentVolumeClaimYaml(relativePath string) (*v1.PersistentVolumeClaim, error) {
	var pvc v1.PersistentVolumeClaim
	marshaled, err := parseYaml(relativePath, "PersistentVolumeClaim")
	if err != nil {
		return nil, err
	}
	json.Unmarshal(marshaled, &pvc)
	return &pvc, nil
}

// ParseVirtualService parse VirtualService from yaml file.
func ParseVirtualService(relativePath string) (*istiov1alpha3.VirtualService, error) {
	var virtualService istiov1alpha3.VirtualService
	marshaled, err := parseYaml(relativePath, "VirtualService")
	if err != nil {
		return nil, err
	}
	json.Unmarshal(marshaled, &virtualService)
	return &virtualService, nil
}

// ParseSecretYaml parse Secret from yaml file.
func ParseSecretYaml(relativePath string) (*v1.Secret, error) {
	var secret v1.Secret
	marshaled, err := parseYaml(relativePath, "Secret")
	if err != nil {
		return nil, err
	}
	json.Unmarshal(marshaled, &secret)
	return &secret, nil
}
