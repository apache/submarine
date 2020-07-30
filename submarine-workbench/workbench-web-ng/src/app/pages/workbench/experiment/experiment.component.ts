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
import { ExperimentFormService } from '@submarine/services/experiment.validator.service';
import { NzMessageService } from 'ng-zorro-antd';
import { SpecMeta, Specs, SpecEnviroment, ExperimentSpec } from '@submarine/interfaces/experiment-spec';
import { stringify } from 'querystring';

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

  // ExperimentSpecs = ['Adhoc', 'Predefined'];
  frameworkNames = ['Tensorflow', 'Pytorch'];

  // About env page
  currentEnvPage = 1;
  PAGESIZE = 5;

  // About spec
  currentSpecPage = 1;

  constructor(
    private experimentService: ExperimentService,
    private experimentFormService: ExperimentFormService,
    private nzMessageService: NzMessageService,
    private route: ActivatedRoute,
    private router: Router
  ) {}

  ngOnInit() {
    this.experiment = new FormGroup({
      experimentName: new FormControl(null, Validators.required),
      description: new FormControl(null, [Validators.required]),
      // experimentSpec: new FormControl('Adhoc'),
      frameworks: new FormControl('Tensorflow', [Validators.required]),
      namespace: new FormControl('default', [Validators.required]),
      // ruleType: new FormControl('Strong'),
      cmd: new FormControl('', [Validators.required]),
      envs: new FormArray([], [this.experimentFormService.nameValidatorFactory('key')]),
      image: new FormControl('', [Validators.required]),
      specs: new FormArray([], [this.experimentFormService.nameValidatorFactory('name')])
    });
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
  get frameworks() {
    return this.experiment.get('frameworks');
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
   * Check the validity of the experiment page
   *
   */
  checkStatus() {
    if (this.current === 0) {
      return (
        this.experimentName.invalid ||
        this.description.invalid ||
        this.frameworks.invalid ||
        this.namespace.invalid ||
        this.cmd.invalid ||
        this.image.invalid
      );
    } else if (this.current === 1) {
      return this.envs.invalid;
    } else if (this.current === 2) {
      return this.specs.invalid;
    }
  }

  /**
   * Init a new experiment form, clear all status
   */
  initExperimentStatus() {
    this.isVisible = false;
    this.current = 0;
    this.okText = 'Next step';
  }

  /**
   * Event handler for Next step/Submit button
   */
  handleOk() {
    if (this.current === 1) {
      this.okText = 'Submit';
    } else if (this.current === 2) {
      const newSpec = this.constructSpec();
      this.experimentService.createExperiment(newSpec).subscribe({
        next: (result) => {
          // Must reconstruct a new array for re-rendering
          this.experimentList = [...this.experimentList, result];
        },
        error: (msg) => {
          this.nzMessageService.error(`${msg}, please try again`, {
            nzPauseOnHover: true
          });
        },
        complete: () => {
          this.nzMessageService.success('Experiment creation succeeds');
          this.initExperimentStatus();
        }
      });
    }

    if (this.current < 2) {
      this.current++;
    }
  }

  /**
   * Create a new env variable input
   */
  createEnvInput() {
    // Create a new FormGroup
    const env = new FormGroup(
      {
        key: new FormControl(''),
        value: new FormControl()
      },
      [this.experimentFormService.envValidator]
    );
    this.envs.push(env);
    // If the new page is created, jump to that page
    if (this.envs.controls.length > 1 && this.envs.controls.length % this.PAGESIZE === 1) {
      this.currentEnvPage += 1;
    }
  }
  /**
   * Create a new spec
   *
   */
  createSpec() {
    const spec = new FormGroup(
      {
        name: new FormControl(''),
        replicas: new FormControl(null, [Validators.min(1)]),
        cpus: new FormControl(null, [Validators.min(1)]),
        memory: new FormControl('', [this.experimentFormService.memoryValidator])
      },
      [this.experimentFormService.specValidator]
    );
    this.specs.push(spec);
    // If the new page is created, jump to that page
    if (this.specs.controls.length > 1 && this.specs.controls.length % this.PAGESIZE === 1) {
      this.currentSpecPage += 1;
    }
  }

  /**
   * Construct spec for new experiment creation
   */
  constructSpec(): ExperimentSpec {
    // Construct the spec
    const meta: SpecMeta = {
      name: this.experimentName.value,
      namespace: this.namespace.value,
      framework: this.frameworks.value,
      cmd: this.cmd.value,
      envVars: {}
    };
    for (const env of this.envs.controls) {
      if (env.get('key').value) {
        meta.envVars[env.get('key').value] = env.get('value').value;
      }
    }

    const specs: Specs = {};
    for (const spec of this.specs.controls) {
      if (spec.get('name').value) {
        specs[spec.get('name').value] = {
          replicas: spec.get('replicas').value,
          resources: `cpu=${spec.get('cpus').value},memory=${spec.get('memory').value}`
        };
      }
    }

    const enviroment: SpecEnviroment = {
      image: this.image.value
    };

    const newExperimentSpec: ExperimentSpec = {
      meta: meta,
      environment: enviroment,
      spec: specs
    };

    return newExperimentSpec;
  }

  /**
   * Delete list items(envs or specs)
   *
   * @param arr - The FormArray containing the item
   * @param index - The index of the item
   */
  deleteItem(arr: FormArray, index: number) {
    arr.removeAt(index);
  }

  fetchExperimentList() {
    this.experimentService.fetchExperimentList().subscribe((list) => {
      this.experimentList = list;
      var currentTime = new Date();
      this.experimentList.forEach((item) => {
        if (item.status == 'Succeeded') {
          var finTime = new Date(item.finishedTime);
          var runTime = new Date(item.runningTime);
          var result = (finTime.getTime() - runTime.getTime()) / 1000;
          item.duration = this.durationHandle(result);
        } else {
          var runTime = new Date(item.runningTime);
          var result = (currentTime.getTime() - runTime.getTime()) / 1000;
          item.duration = this.durationHandle(result);
        }
      });
      this.checkedList = [];
      for (let i = 0; i < this.experimentList.length; i++) {
        this.checkedList.push(false);
      }
    });
  }

  onDeleteExperiment(data: ExperimentInfo, onMessage: boolean) {
    this.experimentService.deleteExperiment(data.experimentId).subscribe(
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
      When reload in info page, ths experimentId will turn into undifined, it will cause breadcrumb miss experimentId. 
      Location.pathname -> /workbench/experiment/info/{experimentID}
      So slice out experimentId string from location.pathname to reassign experimentId.
      */
    if (location.pathname != '/workbench/experiment') {
      var sliceString = new String('/workbench/experiment/info');
      this.experimentID = location.pathname.slice(sliceString.length);
    }
  }

  deleteExperiments() {
    for (let i = this.checkedList.length - 1; i >= 0; i--) {
      if (this.checkedList[i] === true) {
        this.onDeleteExperiment(this.experimentList[i], false);
      }
    }

    this.selectAllChecked = false;
  }

  selectAll() {
    for (let i = 0; i < this.checkedList.length; i++) {
      this.checkedList[i] = this.selectAllChecked;
    }
  }

  durationHandle(secs: number) {
    var hr = Math.floor(secs / 3600);
    var min = Math.floor((secs - hr * 3600) / 60);
    var sec = Math.round(secs) - hr * 3600 - min * 60;
    var showHr;
    var showMin;
    var showSec;
    if (hr < 10) {
      showHr = '0' + hr;
    } else {
      showHr = hr.toString();
    }
    if (min < 10) {
      showMin = '0' + min;
    } else {
      showMin = min.toString();
    }
    if (sec < 10) {
      showSec = '0' + sec;
    } else {
      showSec = sec.toString();
    }
    return showHr + ':' + showMin + ':' + showSec;
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
  // TODO(jasoonn): Edit experiment
  editExperiment(Experiment) {
    console.log(Experiment);
  }
}
