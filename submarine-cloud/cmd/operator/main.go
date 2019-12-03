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

package main

import (
	"fmt"
	clientset "github.com/apache/submarine/submarine-cloud/pkg/client/clientset/versioned"
	informers "github.com/apache/submarine/submarine-cloud/pkg/client/informers/externalversions"
	"github.com/apache/submarine/submarine-cloud/pkg/signals"
	"github.com/golang/glog"
	"github.com/mitchellh/go-homedir"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"os"
	"strings"
	"time"
)

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:   "submarineController",
	Short: "Apache submarine operator",
	Long: "Apache submarine operator",
	// Uncomment the following line if your bare application
	// has an action associated with it:
	//	Run: func(cmd *cobra.Command, args []string) { },
}

var cfgFile string

func init() {
	cobra.OnInitialize(initConfig)

	// Here you will define your flags and configuration settings.
	// Cobra supports persistent flags, which, if defined here,
	// will be global for your application.
	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is $HOME/.samplecontroller.yaml)")

	// Cobra also supports local flags, which will only run
	// when this action is called directly.
	rootCmd.Flags().BoolP("toggle", "t", false, "Help message for toggle")

	rootCmd.AddCommand(runCmd)
	// Here you will define your flags and configuration settings.

	// Cobra supports Persistent Flags which will work for this command
	// and all subcommands, e.g.:
	// runCmd.PersistentFlags().String("foo", "", "A help for foo")

	// Cobra supports local flags which will only run when this command
	// is called directly, e.g.:
	// runCmd.Flags().BoolP("toggle", "t", false, "Help message for toggle")
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

// runCmd represents the run command
var runCmd = &cobra.Command{
	Use:   "run",
	Short: "run config=[kubeConfig's path]",
	Long:  ``,
	Run: func(cmd *cobra.Command, args []string) {
		NewApp(args)
	},
}

// initConfig reads in config file and ENV variables if set.
func initConfig() {
	if cfgFile != "" {
		// Use config file from the flag.
		viper.SetConfigFile(cfgFile)
	} else {
		// Find home directory.
		home, err := homedir.Dir()
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}

		// Search config in home directory with name ".samplecontroller" (without extension).
		viper.AddConfigPath(home)
		viper.SetConfigName(".submarineController")
	}

	viper.AutomaticEnv() // read in environment variables that match

	// If a config file is found, read it in.
	if err := viper.ReadInConfig(); err == nil {
		fmt.Println("Using config file:", viper.ConfigFileUsed())
	}
}

func getKubeCfg(args []string) string {
	for _, v := range args {
		strs := strings.Split(v, "=")
		if strs[0] == "config" {
			return strs[1]
		}
	}
	return ""
}

func getKubeMasterUrl(args []string) string {
	for _, v := range args {
		strs := strings.Split(v, "=")
		if strs[0] == "master" {
			return strs[1]
		}
	}
	return ""
}

func NewApp(args []string) {
	defer glog.Flush()
	// Processing semaphores
	stopCh := signals.SetupSignalHandler()

	// Processing input parameters
	cfg, err := clientcmd.BuildConfigFromFlags(getKubeMasterUrl(args), getKubeCfg(args))
	if err != nil {
		glog.Fatalf("Error building kubeconfig: %s", err.Error())
	}

	// Create a standard client
	kubeClient, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		glog.Fatalf("Error building kubernetes clientset: %s", err.Error())
	}

	// Create a client for student resources
	studentClient, err := clientset.NewForConfig(cfg)
	if err != nil {
		glog.Fatalf("Error building example clientset: %s", err.Error())
	}

	// Create informer
	studentInformerFactory := informers.NewSharedInformerFactory(studentClient, time.Second*30)

	controller := NewSubmarineController(kubeClient, studentClient,
		studentInformerFactory.Submarine().V1alpha1().SubmarineServers())

	// Start informer
	go studentInformerFactory.Start(stopCh)

	// Start controller
	if err = controller.Run(10, stopCh); err != nil {
		glog.Fatalf("Error running controller: %s", err.Error())
	}
}
