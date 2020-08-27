/*!
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

import { Component, OnInit } from '@angular/core';
import { FormArray, FormControl, FormGroup, Validators } from '@angular/forms';
import { ActivatedRoute, NavigationStart, Router } from '@angular/router';
import { ExperimentInfo } from '@submarine/interfaces/experiment-info';
import { ExperimentService } from '@submarine/services/experiment.service';
import { nanoid } from 'nanoid';
import { NzMessageService } from 'ng-zorro-antd';
import { ExperimentSpec } from '@submarine/interfaces/experiment-spec';

@Component({
  selector: 'submarine-experiment',
  templateUrl: './experiment.component.html',
  styleUrls: ['./experiment.component.scss']
})
export class ExperimentComponent implements OnInit {
  experimentList: ExperimentInfo[] = [];
  checkedList: boolean[] = [];
  selectAllChecked: boolean = false;
  // About experiment information
  isInfo = false;
  experimentID: string;

  // About show existing experiments
  showExperiment = 'All';
  searchText = '';

  // About new experiment
  experiment: FormGroup;
  current = 0;
  okText = 'Next Step';
  isVisible = false;

  // About update
  mode: 'create' | 'update' | 'clone' = 'create';
  updateId: string = null;

  statusColor: { [key: string]: string } = {
    Accepted: 'gold',
    Created: 'white',
    Running: 'green',
    Succeeded: 'blue'
  };

  constructor(
    private nzMessageService: NzMessageService,
    private route: ActivatedRoute,
    private router: Router,
    private experimentService: ExperimentService
  ) {}

  ngOnInit() {
    this.fetchExperimentList();
    this.isInfo = this.router.url !== '/workbench/experiment';
    this.experimentID = this.route.snapshot.params.id;
    this.router.events.subscribe((val) => {
      if (val instanceof NavigationStart) {
        console.log(val.url);
        if (val.url === '/workbench/experiment') {
          this.isInfo = false;
          this.fetchExperimentList();
        } else {
          this.isInfo = true;
        }
      }
    });

    this.reloadCheck();
  }

  // Getters of experiment request form
  get experimentName() {
    return this.experiment.get('experimentName');
  }
  get description() {
    return this.experiment.get('description');
  }
  get namespace() {
    return this.experiment.get('namespace');
  }
  get cmd() {
    return this.experiment.get('cmd');
  }
  get envs() {
    return this.experiment.get('envs') as FormArray;
  }
  get image() {
    return this.experiment.get('image');
  }
  get specs() {
    return this.experiment.get('specs') as FormArray;
  }
  /**
   * Init a new experiment form, clear all status, clear all form controls and open the form in the mode specified in the argument
   *
   * @param mode - The mode which the form should open in
   */
  initExperimentStatus(mode: 'create' | 'update' | 'clone') {
    this.mode = mode;
    this.current = 0;
    this.okText = 'Next step';
    this.isVisible = true;
    this.updateId = null;
    // Reset the form
    this.experimentName.enable();
    this.envs.clear();
    this.specs.clear();
    this.experiment.reset({ frameworks: 'TensorFlow', namespace: 'default' });
  }
  /**
   * Check the validity of the experiment page
   *
   */
  checkStatus() {
    if (this.current === 0) {
      // return (
      //   this.experimentName.invalid ||
      //   this.namespace.invalid ||
      //   this.cmd.invalid ||
      //   this.image.invalid

      // );
      return false;
    } else if (this.current === 1) {
      return this.envs.invalid;
    } else if (this.current === 2) {
      return this.specs.invalid;
    }
  }
  /**
   * Event handler for Next step/Submit button
   */
  // handleOk() {
  //   if (this.current === 1) {
  //     this.okText = 'Submit';
  //   } else if (this.current === 2) {
  //     if (this.mode === 'create') {
  //       const newSpec = this.constructSpec();
  //       this.experimentService.createExperiment(newSpec).subscribe({
  //         next: (result) => {
  //           this.fetchExperimentList();
  //         },
  //         error: (msg) => {
  //           this.nzMessageService.error(`${msg}, please try again`, {
  //             nzPauseOnHover: true
  //           });
  //         },
  //         complete: () => {
  //           this.nzMessageService.success('Experiment creation succeeds');
  //           this.isVisible = false;
  //         }
  //       });
  //     } else if (this.mode === 'update') {
  //       const newSpec = this.constructSpec();
  //       this.experimentService.updateExperiment(this.updateId, newSpec).subscribe(
  //         () => {
  //           this.fetchExperimentList();
  //         },
  //         (msg) => {
  //           this.nzMessageService.error(`${msg}, please try again`, {
  //             nzPauseOnHover: true
  //           });
  //         },
  //         () => {
  //           this.nzMessageService.success('Modification succeeds!');
  //           this.isVisible = false;
  //         }
  //       );
  //     } else if (this.mode === 'clone') {
  //       const newSpec = this.constructSpec();
  //       this.experimentService.createExperiment(newSpec).subscribe(
  //         () => {
  //           this.fetchExperimentList();
  //         },
  //         (msg) => {
  //           this.nzMessageService.error(`${msg}, please try again`, {
  //             nzPauseOnHover: true
  //           });
  //         },
  //         () => {
  //           this.nzMessageService.success('Create a new experiment !');
  //           this.isVisible = false;
  //         }
  //       );
  //     }
  //   }

  //   if (this.current < 2) {
  //     this.current++;
  //   }
  // }

  fetchExperimentList() {
    this.experimentService.fetchExperimentList().subscribe((list) => {
      this.experimentList = list;
      const currentTime = new Date();
      this.experimentList.forEach((item) => {
        if (item.status === 'Succeeded') {
          const finTime = new Date(item.finishedTime);
          const runTime = new Date(item.runningTime);
          const result = (finTime.getTime() - runTime.getTime()) / 1000;
          item.duration = this.experimentService.durationHandle(result);
        } else if (item.runningTime) {
          const runTime = new Date(item.runningTime);
          const result = (currentTime.getTime() - runTime.getTime()) / 1000;
          item.duration = this.experimentService.durationHandle(result);
        }
      });
      this.checkedList = [];
      for (let i = 0; i < this.experimentList.length; i++) {
        this.checkedList.push(false);
      }
    });
  }

  onUpdateExperiment(id: string, spec: ExperimentSpec) {
    // Open Modal in update mode
    this.initExperimentStatus('update');
    // Keep id for later request
    this.updateId = id;

    // Prevent user from modifying the name
    this.experimentName.disable();

    // Put value back
    this.experimentName.setValue(spec.meta.name);
    this.cloneExperiment(spec);
  }

  onCloneExperiment(spec: ExperimentSpec) {
    // Open Modal in update mode
    this.initExperimentStatus('clone');
    // Prevent user from modifying the name
    this.experimentName.enable();
    // Put value back
    const id: string = nanoid(8);
    const cloneExperimentName = spec.meta.name + '-' + id;
    this.experimentName.setValue(cloneExperimentName.toLocaleLowerCase());
    this.cloneExperiment(spec);
  }

  cloneExperiment(spec: ExperimentSpec) {
    // this.description.setValue(spec.meta.description);
    // this.namespace.setValue(spec.meta.namespace);
    // this.cmd.setValue(spec.meta.cmd);
    // this.image.setValue(spec.environment.image);
    // for (const [key, value] of Object.entries(spec.meta.envVars)) {
    //   const env = this.createEnv(key, value);
    //   this.envs.push(env);
    // }
    // for (const [specName, info] of Object.entries(spec.spec)) {
    //   const [cpuCount, memory, unit] = info.resources.match(/\d+|[MG]/g);
    //   const newSpec = this.createSpec(specName, parseInt(info.replicas, 10), parseInt(cpuCount, 10), memory, unit);
    //   this.specs.push(newSpec);
    // }
  }

  onDeleteExperiment(id: string, onMessage: boolean) {
    this.experimentService.deleteExperiment(id).subscribe(
      () => {
        if (onMessage === true) {
          this.nzMessageService.success('Delete Experiment Successfully!');
        }
        this.fetchExperimentList();
      },
      (err) => {
        if (onMessage === true) {
          this.nzMessageService.success(err.message);
        }
      }
    );
  }

  reloadCheck() {
    /*
      When reload in info page, ths experimentId will turn into undefined, it will cause breadcrumb miss experimentId.
      Location.pathname -> /workbench/experiment/info/{experimentID}
      So slice out experimentId string from location.pathname to reassign experimentId.
      */
    if (location.pathname !== '/workbench/experiment') {
      const sliceString = String('/workbench/experiment/info');
      this.experimentID = location.pathname.slice(sliceString.length + 1);
    }
  }

  deleteExperiments() {
    for (let i = this.checkedList.length - 1; i >= 0; i--) {
      if (this.checkedList[i] === true) {
        this.onDeleteExperiment(this.experimentList[i].experimentId, false);
      }
    }

    this.selectAllChecked = false;
  }

  selectAll() {
    for (let i = 0; i < this.checkedList.length; i++) {
      this.checkedList[i] = this.selectAllChecked;
    }
  }

  // TODO(jasoonn): Filter experiment list
  filter(event) {
    console.log(this.searchText + event.key);
  }
  // TODO(jasoonn): Perform part of list
  showChange() {
    console.log('Change to ' + this.showExperiment);
  }
  // TODO(jasoonn): Start experiment
  startExperiment(Experiment) {
    console.log(Experiment);
  }
}
