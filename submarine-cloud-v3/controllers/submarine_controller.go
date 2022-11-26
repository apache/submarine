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
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/go-logr/logr"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	submarineapacheorgv1alpha1 "github.com/apache/submarine/submarine-cloud-v3/api/v1alpha1"
)

// Defines resource names and path to artifact yaml files
// Reference: https://github.com/apache/submarine/blob/master/submarine-cloud-v3/artifacts/
const (
	serverName             = "submarine-server"
	observerName           = "submarine-observer"
	databaseName           = "submarine-database"
	tensorboardName        = "submarine-tensorboard"
	mlflowName             = "submarine-mlflow"
	minioName              = "submarine-minio"
	storageName            = "submarine-storage"
	virtualServiceName     = "submarine-virtual-service"
	databasePvcName        = databaseName + "-pvc"
	tensorboardPvcName     = tensorboardName + "-pvc"
	tensorboardServiceName = tensorboardName + "-service"
	mlflowPvcName          = mlflowName + "-pvc"
	mlflowServiceName      = mlflowName + "-service"
	minioPvcName           = minioName + "-pvc"
	minioServiceName       = minioName + "-service"
	artifactPath           = "./artifacts/"
	databaseYamlPath       = artifactPath + "submarine-database.yaml"
	minioYamlPath          = artifactPath + "submarine-minio.yaml"
	serveYamlPath          = artifactPath + "submarine-serve.yaml"
	mlflowYamlPath         = artifactPath + "submarine-mlflow.yaml"
	serverYamlPath         = artifactPath + "submarine-server.yaml"
	tensorboardYamlPath    = artifactPath + "submarine-tensorboard.yaml"
	serverRbacYamlPath     = artifactPath + "submarine-server-rbac.yaml"
	observerRbacYamlPath   = artifactPath + "submarine-observer-rbac.yaml"
	storageRbacYamlPath    = artifactPath + "submarine-storage-rbac.yaml"
	virtualServiceYamlPath = artifactPath + "submarine-virtualservice.yaml"
)

// Name of deployments whose replica count and readiness need to be checked
var dependents = []string{serverName, tensorboardName, mlflowName, minioName}

const (
	// SuccessSynced is used as part of the Event 'reason' when a Submarine is synced
	//SuccessSynced = "Synced"

	// MessageResourceSynced is the message used for an Event fired when a
	// Submarine is synced successfully
	//MessageResourceSynced = "Submarine synced successfully"

	// ErrResourceExists is used as part of the Event 'reason' when a Submarine fails
	// to sync due to a Deployment of the same name already existing.
	ErrResourceExists = "ErrResourceExists"

	// MessageResourceExists is the message used for Events when a resource
	// fails to sync due to a Deployment already existing
	MessageResourceExists = "Resource %q already exists and is not managed by Submarine"
)

// Default k8s anyuid role rule
var k8sAnyuidRoleRule = rbacv1.PolicyRule{
	APIGroups:     []string{"policy"},
	Verbs:         []string{"use"},
	Resources:     []string{"podsecuritypolicies"},
	ResourceNames: []string{"submarine-anyuid"},
}

// Openshift anyuid role rule
var openshiftAnyuidRoleRule = rbacv1.PolicyRule{
	APIGroups:     []string{"security.openshift.io"},
	Verbs:         []string{"use"},
	Resources:     []string{"securitycontextconstraints"},
	ResourceNames: []string{"anyuid"},
}

// SubmarineReconciler reconciles a Submarine object
type SubmarineReconciler struct {
	// Fields required by the operator
	client.Client
	Scheme   *runtime.Scheme
	Log      logr.Logger
	Recorder record.EventRecorder
	// Fields required by submarine
	ClusterType             string
	CreatePodSecurityPolicy bool
}

// kubebuilder RBAC markers generates rules for the operator ClusterRole
// On change, run `make manifest` to update config/rbac/role.yaml
// Reference: https://github.com/apache/submarine/blob/master/submarine-cloud-v3/config/rbac/role.yaml

// Submarine resources
//+kubebuilder:rbac:groups=submarine.apache.org,resources=submarines,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=submarine.apache.org,resources=submarines/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=submarine.apache.org,resources=submarines/finalizers,verbs=update

// Event
//+kubebuilder:rbac:groups="",resources=events,verbs=create;patch
//+kubebuilder:rbac:groups="",resources=pods;secrets;configmaps;namespaces;jobs,verbs="*"
//+kubebuilder:rbac:groups="apiextensions.k8s.io",resources=customresourcedefinitions,verbs="*"

// k8s resources
//+kubebuilder:rbac:groups=apps,resources=deployments;statefulsets,verbs="*"
//+kubebuilder:rbac:groups=apps,resources=replicasets,verbs="*"
//+kubebuilder:rbac:groups=core,resources=persistentvolumeclaims;serviceaccounts;services,verbs="*"
//+kubebuilder:rbac:groups=rbac.authorization.k8s.io,resources=roles;rolebindings,verbs="*"

// kubeflow resources
//+kubebuilder:rbac:groups=kubeflow.org,resources=notebooks;pytorchjobs;tfjobs;xgboostjobs,verbs="*"

// Istio resources
//+kubebuilder:rbac:groups=networking.istio.io,resources=virtualservices,verbs="*"

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.11.0/pkg/reconcile
func (r *SubmarineReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	r.Log.Info("Enter Reconcile", "req", req)

	// Get the Submarine resource with the requested name/namespace
	submarine := &submarineapacheorgv1alpha1.Submarine{}
	err := r.Get(ctx, types.NamespacedName{Name: req.Name, Namespace: req.Namespace}, submarine)
	if err != nil {
		if errors.IsNotFound(err) {
			// The Submarine resource may no longer exist, in which case we stop processing
			r.Log.Error(nil, "Submarine no longer exists", "name", req.Name, "namespace", req.Namespace)
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// Submarine is in the terminating process, only used when in foreground cascading deletion, otherwise the submarine will be recreated
	if !submarine.DeletionTimestamp.IsZero() {
		return ctrl.Result{}, nil
	}

	submarineCopy := submarine.DeepCopy()

	// Take action based on submarine state
	// State machine for Submarine:
	//+-----------------------------------------------------------------+
	//|      +---------+         +----------+          +----------+     |
	//|      |         |         |          |          |          |     |
	//|      |   New   +---------> Creating +----------> Running  |     |
	//|      |         |         |          |          |          |     |
	//|      +----+----+         +-----+----+          +-----+----+     |
	//|           |                    |                     |          |
	//|           |                    |                     |          |
	//|           |                    |                     |          |
	//|           |                    |               +-----v----+     |
	//|           |                    |               |          |     |
	//|           +--------------------+--------------->  Failed  |     |
	//|                                                |          |     |
	//|                                                +----------+     |
	//+-----------------------------------------------------------------+
	switch submarineCopy.Status.State {
	case submarineapacheorgv1alpha1.NewState:
		r.recordSubmarineEvent(submarineCopy)
		if err := r.validateSubmarine(submarineCopy); err != nil {
			submarineCopy.Status.State = submarineapacheorgv1alpha1.FailedState
			submarineCopy.Status.ErrorMessage = err.Error()
			r.recordSubmarineEvent(submarineCopy)
		} else {
			submarineCopy.Status.State = submarineapacheorgv1alpha1.CreatingState
			r.recordSubmarineEvent(submarineCopy)
		}
	case submarineapacheorgv1alpha1.CreatingState:
		if err := r.createSubmarine(ctx, submarineCopy); err != nil {
			submarineCopy.Status.State = submarineapacheorgv1alpha1.FailedState
			submarineCopy.Status.ErrorMessage = err.Error()
			r.recordSubmarineEvent(submarineCopy)
		}
		ok, err := r.checkSubmarineDependentsReady(ctx, submarineCopy)
		if err != nil {
			submarineCopy.Status.State = submarineapacheorgv1alpha1.FailedState
			submarineCopy.Status.ErrorMessage = err.Error()
			r.recordSubmarineEvent(submarineCopy)
		}
		if ok {
			submarineCopy.Status.State = submarineapacheorgv1alpha1.RunningState
			r.recordSubmarineEvent(submarineCopy)
		}
	case submarineapacheorgv1alpha1.RunningState:
		if err := r.createSubmarine(ctx, submarineCopy); err != nil {
			submarineCopy.Status.State = submarineapacheorgv1alpha1.FailedState
			submarineCopy.Status.ErrorMessage = err.Error()
			r.recordSubmarineEvent(submarineCopy)
		}
	}

	// Update STATUS of Submarine
	err = r.updateSubmarineStatus(ctx, submarine, submarineCopy)
	if err != nil {
		return ctrl.Result{}, err
	}

	// Re-run Reconcile regularly
	result := ctrl.Result{}
	result.RequeueAfter = time.Second * 30 // default resync period
	return result, nil
}

func (r *SubmarineReconciler) updateSubmarineStatus(ctx context.Context, submarine, submarineCopy *submarineapacheorgv1alpha1.Submarine) error {
	// Update server replicas
	serverDeployment := &appsv1.Deployment{}
	err := r.Get(ctx, types.NamespacedName{Name: serverName, Namespace: submarine.Namespace}, serverDeployment)
	if err != nil {
		if errors.IsNotFound(err) {
			submarineCopy.Status.AvailableServerReplicas = serverDeployment.Status.AvailableReplicas
		} else {
			return err
		}
	}

	// Update database replicas
	statefulset := &appsv1.StatefulSet{}
	err = r.Get(ctx, types.NamespacedName{Name: databaseName, Namespace: submarine.Namespace}, statefulset)
	if err != nil {
		if errors.IsNotFound(err) {
			submarineCopy.Status.AvailableDatabaseReplicas = statefulset.Status.ReadyReplicas
		} else {
			return err
		}
	}

	// Skip update if nothing changed.
	if equality.Semantic.DeepEqual(submarine.Status, submarineCopy.Status) {
		return nil
	}

	// Update submarine status
	err = r.Status().Update(ctx, submarineCopy)
	if err != nil {
		return err
	}
	return nil
}

func (r *SubmarineReconciler) validateSubmarine(submarine *submarineapacheorgv1alpha1.Submarine) error {
	// Print out the spec of the Submarine resource
	b, err := json.MarshalIndent(submarine.Spec, "", "  ")
	fmt.Println(string(b))

	if err != nil {
		return err
	}

	return nil
}

// Creates resources according to artifact yaml files
// Reference: https://github.com/apache/submarine/blob/master/submarine-cloud-v3/artifacts/
func (r *SubmarineReconciler) createSubmarine(ctx context.Context, submarine *submarineapacheorgv1alpha1.Submarine) error {
	var err error
	// We create rbac first, this ensures that any dependency based on it will not go wrong
	err = r.createSubmarineServerRBAC(ctx, submarine)
	if err != nil && !errors.IsAlreadyExists(err) {
		return err
	}

	err = r.createSubmarineStorageRBAC(ctx, submarine)
	if err != nil && !errors.IsAlreadyExists(err) {
		return err
	}

	err = r.createSubmarineObserverRBAC(ctx, submarine)
	if err != nil && !errors.IsAlreadyExists(err) {
		return err
	}

	err = r.createSubmarineServer(ctx, submarine)
	if err != nil && !errors.IsAlreadyExists(err) {
		return err
	}

	err = r.createSubmarineDatabase(ctx, submarine)
	if err != nil && !errors.IsAlreadyExists(err) {
		return err
	}

	err = r.createVirtualService(ctx, submarine)
	if err != nil && !errors.IsAlreadyExists(err) {
		return err
	}

	err = r.createSubmarineTensorboard(ctx, submarine)
	if err != nil && !errors.IsAlreadyExists(err) {
		return err
	}

	err = r.createSubmarineMlflow(ctx, submarine)
	if err != nil && !errors.IsAlreadyExists(err) {
		return err
	}

	err = r.createSubmarineMinio(ctx, submarine)
	if err != nil && !errors.IsAlreadyExists(err) {
		return err
	}

	err = r.createSubmarineServe(ctx, submarine)
	if err != nil && !errors.IsAlreadyExists(err) {
		return err
	}

	return nil
}

// Checks the number of deployment and database replicas and if they are ready
func (r *SubmarineReconciler) checkSubmarineDependentsReady(ctx context.Context, submarine *submarineapacheorgv1alpha1.Submarine) (bool, error) {
	// deployment dependents check
	for _, name := range dependents {
		// 1. Check if deployment exists
		deployment := &appsv1.Deployment{}
		err := r.Get(ctx, types.NamespacedName{Name: name, Namespace: submarine.Namespace}, deployment)
		if err != nil {
			if errors.IsNotFound(err) {
				return false, nil
			}
			return false, err
		}
		// 2. Check if deployment replicas failed
		for _, condition := range deployment.Status.Conditions {
			if condition.Type == appsv1.DeploymentReplicaFailure {
				return false, fmt.Errorf("failed creating replicas of %s, message: %s", deployment.Name, condition.Message)
			}
		}
		// 3. Check if ready replicas are same as targeted replicas
		if deployment.Status.ReadyReplicas != deployment.Status.Replicas {
			return false, nil
		}
	}
	// database check
	// 1. Check if database exists
	statefulset := &appsv1.StatefulSet{}
	err := r.Get(ctx, types.NamespacedName{Name: databaseName, Namespace: submarine.Namespace}, statefulset)
	if err != nil {
		if errors.IsNotFound(err) {
			return false, nil
		}
		return false, err
	}

	// 2. Check if database replicas failed
	// statefulset.Status.Conditions does not have the specified type enum like
	// deployment.Status.Conditions => DeploymentConditionType ,
	// so we will ignore the verification status for the time being

	// 3. Check if ready replicas are same as targeted replicas
	if statefulset.Status.Replicas != statefulset.Status.ReadyReplicas {
		return false, nil
	}

	return true, nil
}

// Wraps r.Recorder.Eventf
// Fill reason, message fields of Event according to state of Submarine
func (r *SubmarineReconciler) recordSubmarineEvent(submarine *submarineapacheorgv1alpha1.Submarine) {
	switch submarine.Status.State {
	case submarineapacheorgv1alpha1.NewState:
		r.Recorder.Eventf(
			submarine,
			corev1.EventTypeNormal,
			"SubmarineAdded",
			"Submarine %s was added",
			submarine.Name)
	case submarineapacheorgv1alpha1.CreatingState:
		r.Recorder.Eventf(
			submarine,
			corev1.EventTypeNormal,
			"SubmarineCreating",
			"Submarine %s was creating",
			submarine.Name,
		)
	case submarineapacheorgv1alpha1.RunningState:
		r.Recorder.Eventf(
			submarine,
			corev1.EventTypeNormal,
			"SubmarineRunning",
			"Submarine %s was running",
			submarine.Name,
		)
	case submarineapacheorgv1alpha1.FailedState:
		r.Recorder.Eventf(
			submarine,
			corev1.EventTypeWarning,
			"SubmarineFailed",
			"Submarine %s was failed: %s",
			submarine.Name,
			submarine.Status.SubmarineState.ErrorMessage,
		)
	}
}

// SetupWithManager sets up the controller with the Manager.
func (r *SubmarineReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&submarineapacheorgv1alpha1.Submarine{}).
		Complete(r)
}

// CreatePullSecrets will convert `submarine.spec.common.image.pullSecrets` to []`corev1.LocalObjectReference`
func (r *SubmarineReconciler) CreatePullSecrets(pullSecrets *[]string) []corev1.LocalObjectReference {
	secrets := make([]corev1.LocalObjectReference, 0)
	for _, secret := range *pullSecrets {
		secrets = append(secrets, corev1.LocalObjectReference{Name: secret})
	}
	return secrets
}
