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
	clientset "github.com/apache/submarine/submarine-cloud-v2/pkg/client/clientset/versioned"
	informers "github.com/apache/submarine/submarine-cloud-v2/pkg/client/informers/externalversions/submarine/v1alpha1"
	traefik "github.com/traefik/traefik/v2/pkg/provider/kubernetes/crd/generated/clientset/versioned"
	traefikinformers "github.com/traefik/traefik/v2/pkg/provider/kubernetes/crd/generated/informers/externalversions/traefik/v1alpha1"
	appsinformers "k8s.io/client-go/informers/apps/v1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	extinformers "k8s.io/client-go/informers/extensions/v1beta1"
	rbacinformers "k8s.io/client-go/informers/rbac/v1"
	"k8s.io/client-go/kubernetes"
)

type BuilderConfig struct {
	incluster                     bool
	kubeclientset                 kubernetes.Interface
	submarineclientset            clientset.Interface
	traefikclientset              traefik.Interface
	namespaceInformer             coreinformers.NamespaceInformer
	deploymentInformer            appsinformers.DeploymentInformer
	serviceInformer               coreinformers.ServiceInformer
	serviceaccountInformer        coreinformers.ServiceAccountInformer
	persistentvolumeclaimInformer coreinformers.PersistentVolumeClaimInformer
	ingressInformer               extinformers.IngressInformer
	ingressrouteInformer          traefikinformers.IngressRouteInformer
	roleInformer                  rbacinformers.RoleInformer
	rolebindingInformer           rbacinformers.RoleBindingInformer
	submarineInformer             informers.SubmarineInformer
}

func NewControllerBuilderConfig() *BuilderConfig {
	return &BuilderConfig{}
}

func (bc *BuilderConfig) InCluster(
	incluster bool,
) *BuilderConfig {
	bc.incluster = incluster
	return bc
}

func (bc *BuilderConfig) WithKubeClientset(
	kubeclientset kubernetes.Interface,
) *BuilderConfig {
	bc.kubeclientset = kubeclientset
	return bc
}

func (bc *BuilderConfig) WithSubmarineClientset(
	submarineclientset clientset.Interface,
) *BuilderConfig {
	bc.submarineclientset = submarineclientset
	return bc
}

func (bc *BuilderConfig) WithTraefikClientset(
	traefikclientset traefik.Interface,
) *BuilderConfig {
	bc.traefikclientset = traefikclientset
	return bc
}

func (bc *BuilderConfig) WithSubmarineInformer(
	submarineInformer informers.SubmarineInformer,
) *BuilderConfig {
	bc.submarineInformer = submarineInformer
	return bc
}

func (bc *BuilderConfig) WithNamespaceInformer(
	namespaceInformer coreinformers.NamespaceInformer,
) *BuilderConfig {
	bc.namespaceInformer = namespaceInformer
	return bc
}

func (bc *BuilderConfig) WithDeploymentInformer(
	deploymentInformer appsinformers.DeploymentInformer,
) *BuilderConfig {
	bc.deploymentInformer = deploymentInformer
	return bc
}

func (bc *BuilderConfig) WithServiceInformer(
	serviceInformer coreinformers.ServiceInformer,
) *BuilderConfig {
	bc.serviceInformer = serviceInformer
	return bc
}

func (bc *BuilderConfig) WithServiceAccountInformer(
	serviceaccountInformer coreinformers.ServiceAccountInformer,
) *BuilderConfig {
	bc.serviceaccountInformer = serviceaccountInformer
	return bc
}

func (bc *BuilderConfig) WithPersistentVolumeClaimInformer(
	persistentvolumeclaimInformer coreinformers.PersistentVolumeClaimInformer,
) *BuilderConfig {
	bc.persistentvolumeclaimInformer = persistentvolumeclaimInformer
	return bc
}

func (bc *BuilderConfig) WithIngressInformer(
	ingressInformer extinformers.IngressInformer,
) *BuilderConfig {
	bc.ingressInformer = ingressInformer
	return bc
}

func (bc *BuilderConfig) WithIngressRouteInformer(
	ingressrouteInformer traefikinformers.IngressRouteInformer,
) *BuilderConfig {
	bc.ingressrouteInformer = ingressrouteInformer
	return bc
}

func (bc *BuilderConfig) WithRoleInformer(
	roleInformer rbacinformers.RoleInformer,
) *BuilderConfig {
	bc.roleInformer = roleInformer
	return bc
}

func (bc *BuilderConfig) WithRoleBindingInformer(
	rolebindingInformer rbacinformers.RoleBindingInformer,
) *BuilderConfig {
	bc.rolebindingInformer = rolebindingInformer
	return bc
}
