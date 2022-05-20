/*
Copyright 2022.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controllers

import (
	"context"

	"github.com/go-logr/logr"

	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

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

// Submarine resource
//+kubebuilder:rbac:groups=submarine.apache.org,resources=submarines,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=submarine.apache.org,resources=submarines/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=submarine.apache.org,resources=submarines/finalizers,verbs=update

// Event
//+kubebuilder:rbac:groups="",resources=events,verbs=create;patch

// Other resources
//+kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=services,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=serviceaccounts,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=persistentvolumeclaims,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=apps,resources=statefulsets,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=rbac,resources=roles,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=rbac,resources=rolebindings,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=networking.istio.io,resources=virtualservices,verbs=get;list;watch;create;update;patch;delete

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the Submarine object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.11.0/pkg/reconcile
func (r *SubmarineReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	_ = log.FromContext(ctx)

	// TODO(user): your logic here

	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *SubmarineReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&submarineapacheorgv1alpha1.Submarine{}).
		Complete(r)
}
