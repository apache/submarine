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
import { ExperimentSpec, Specs, SpecEnviroment, SpecMeta } from '@submarine/interfaces/experiment-spec';
import { ExperimentService } from '@submarine/services/experiment.service';
import { ExperimentFormService } from '@submarine/services/experiment.validator.service';
import { nanoid } from 'nanoid';
import { NzMessageService } from 'ng-zorro-antd';

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

  FRAMEWORK_NAMES = ['TensorFlow', 'PyTorch'];
  TF_SPECNAMES = ['Master', 'Worker', 'Ps'];
  PYTORCH_SPECNAMES = ['Master', 'Worker'];
  MEMORY_UNITS = ['M', 'G'];

  // About env page
  currentEnvPage = 1;
  PAGESIZE = 5;

  // About spec
  currentSpecPage = 1;

  statusColor: { [key: string]: string } = {
    Accepted: 'gold',
    Created: 'white',
    Running: 'green',
    Succeeded: 'blue'
  };

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
      frameworks: new FormControl('TensorFlow', [Validators.required]),
      namespace: new FormControl('default', [Validators.required]),
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
   * Event handler for Next step/Submit button
   */
  handleOk() {
    if (this.current === 1) {
      this.okText = 'Submit';
    } else if (this.current === 2) {
      if (this.mode === 'create') {
        const newSpec = this.constructSpec();
        this.experimentService.createExperiment(newSpec).subscribe({
          next: (result) => {
            this.fetchExperimentList();
          },
          error: (msg) => {
            this.nzMessageService.error(`${msg}, please try again`, {
              nzPauseOnHover: true
            });
          },
          complete: () => {
            this.nzMessageService.success('Experiment creation succeeds');
            this.isVisible = false;
          }
        });
      } else if (this.mode === 'update') {
        const newSpec = this.constructSpec();
        this.experimentService.updateExperiment(this.updateId, newSpec).subscribe(
          () => {
            this.fetchExperimentList();
          },
          (msg) => {
            this.nzMessageService.error(`${msg}, please try again`, {
              nzPauseOnHover: true
            });
          },
          () => {
            this.nzMessageService.success('Modification succeeds!');
            this.isVisible = false;
          }
        );
      } else if (this.mode === 'clone') {
        const newSpec = this.constructSpec();
        this.experimentService.createExperiment(newSpec).subscribe(
          () => {
            this.fetchExperimentList();
          },
          (msg) => {
            this.nzMessageService.error(`${msg}, please try again`, {
              nzPauseOnHover: true
            });
          },
          () => {
            this.nzMessageService.success('Create a new experiment !');
            this.isVisible = false;
          }
        );
      }
    }

    if (this.current < 2) {
      this.current++;
    }
  }

  /**
   * Create a new env variable input
   */
  createEnv(defaultKey: string = '', defaultValue: string = '') {
    // Create a new FormGroup
    return new FormGroup(
      {
        key: new FormControl(defaultKey, [Validators.required]),
        value: new FormControl(defaultValue, [Validators.required])
      },
      [this.experimentFormService.envValidator]
    );
  }
  /**
   * Create a new spec
   */
  createSpec(
    defaultName: string = '',
    defaultReplica: number = 1,
    defaultCpu: number = 1,
    defaultMemory: string = '',
    defaultUnit: string = 'M'
  ): FormGroup {
    return new FormGroup(
      {
        name: new FormControl(defaultName, [Validators.required]),
        replicas: new FormControl(defaultReplica, [Validators.min(1), Validators.required]),
        cpus: new FormControl(defaultCpu, [Validators.min(1), Validators.required]),
        memory: new FormGroup(
          {
            num: new FormControl(defaultMemory, [Validators.required]),
            unit: new FormControl(defaultUnit, [Validators.required])
          },
          [this.experimentFormService.memoryValidator]
        )
      },
      [this.experimentFormService.specValidator]
    );
  }

  /**
   * Handler for the create env button
   */
  onCreateEnv() {
    const env = this.createEnv();
    this.envs.push(env);
    // If the new page is created, jump to that page
    if (this.envs.controls.length > 1 && this.envs.controls.length % this.PAGESIZE === 1) {
      this.currentEnvPage += 1;
    }
  }

  /**
   * Handler for the create spec button
   */
  onCreateSpec() {
    const spec = this.createSpec();
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
          resources: `cpu=${spec.get('cpus').value},memory=${spec.get('memory').get('num').value}${
            spec.get('memory').get('unit').value
          }`
        };
      }
    }

    const environment: SpecEnviroment = {
      image: this.image.value
    };

    const newExperimentSpec: ExperimentSpec = {
      meta: meta,
      environment: environment,
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
    this.description.setValue(spec.meta.description);
    this.namespace.setValue(spec.meta.namespace);
    this.cmd.setValue(spec.meta.cmd);
    this.image.setValue(spec.environment.image);

    for (const [key, value] of Object.entries(spec.meta.envVars)) {
      const env = this.createEnv(key, value);
      this.envs.push(env);
    }

    for (const [specName, info] of Object.entries(spec.spec)) {
      const [cpuCount, memory, unit] = info.resources.match(/\d+|[MG]/g);
      const newSpec = this.createSpec(specName, parseInt(info.replicas, 10), parseInt(cpuCount, 10), memory, unit);
      this.specs.push(newSpec);
    }
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
          this.nzMessageService.error(err.message);
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
