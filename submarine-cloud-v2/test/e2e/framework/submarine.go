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

package framework

import (
	"context"
	"fmt"

	v1alpha1 "github.com/apache/submarine/submarine-cloud-v2/pkg/apis/submarine/v1alpha1"
	clientset "github.com/apache/submarine/submarine-cloud-v2/pkg/client/clientset/versioned"

	"github.com/pkg/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/yaml"
)

func MakeSubmarineFromYaml(pathToYaml string) (*v1alpha1.Submarine, error) {
	manifest, err := PathToOSFile(pathToYaml)
	if err != nil {
		return nil, err
	}
	tmp := v1alpha1.Submarine{}
	if err := yaml.NewYAMLOrJSONDecoder(manifest, 100).Decode(&tmp); err != nil {
		return nil, errors.Wrap(err, fmt.Sprintf("failed to decode file %s", pathToYaml))
	}
	return &tmp, err
}

func CreateSubmarine(clientset clientset.Interface, namespace string, submarine *v1alpha1.Submarine) error {
	_, err := clientset.SubmarineV1alpha1().Submarines(namespace).Create(context.TODO(), submarine, metav1.CreateOptions{})
	if err != nil {
		return errors.Wrap(err, fmt.Sprintf("failed to create Submarine %s", submarine.Name))
	}
	return nil
}

func UpdateSubmarine(clientset clientset.Interface, namespace string, submarine *v1alpha1.Submarine) error {
	_, err := clientset.SubmarineV1alpha1().Submarines(namespace).Update(context.TODO(), submarine, metav1.UpdateOptions{})
	if err != nil {
		return errors.Wrap(err, fmt.Sprintf("failed to update Submarine %s", submarine.Name))
	}
	return nil
}

func GetSubmarine(clientset clientset.Interface, namespace string, name string) (*v1alpha1.Submarine, error) {
	submarine, err := clientset.SubmarineV1alpha1().Submarines(namespace).Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	return submarine, nil
}

func DeleteSubmarine(clientset clientset.Interface, namespace string, name string) error {
	err := clientset.SubmarineV1alpha1().Submarines(namespace).Delete(context.TODO(), name, metav1.DeleteOptions{})
	if err != nil {
		return err
	}
	return nil
}
