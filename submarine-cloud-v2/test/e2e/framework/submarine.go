package framework

import (
	"context"
	"fmt"
	clientset "submarine-cloud-v2/pkg/generated/clientset/versioned"
	v1alpha1 "submarine-cloud-v2/pkg/submarine/v1alpha1"

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
	submarine, err := clientset.SubmarineV1alpha1().Submarines(namespace).Get(context.TODO(), name, metav1.GetOptinos{})
	if err != nil {
		return nil, err
	}
	return submarine, nil
}

func DeleteSubmarine(clientset clientset.Interface, namespace string, submarine *v1alpha1.Submarine) error {
	_, err := clientset.SubmarineV1alpha1().Submarines(namespace).Delete(context.TODO(), submarine, metav1.DeleteOptions{})
	if err != nil {
		return err
	}
	return nil
}
