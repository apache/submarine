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
import { FormControl, FormGroup, Validators, FormArray } from '@angular/forms';
import { ActivatedRoute, NavigationStart, Router } from '@angular/router';
import { ExperimentInfo } from '@submarine/interfaces/experiment-info';
import { ExperimentService } from '@submarine/services/experiment.service';
import { ExperimentFormService } from '@submarine/services/experiment.validator.service';
import { NzMessageService } from 'ng-zorro-antd';


@Component({
  selector: 'submarine-experiment',
  templateUrl: './experiment.component.html',
  styleUrls: ['./experiment.component.scss']
})
export class ExperimentComponent implements OnInit {
  experimentList: ExperimentInfo[] = [];
  // About experiment information
  isInfo = false;
  experimentID: string;

  // About show existing experiments
  showExperiment = 'All';
  searchText = '';

  // About new experiment
  createExperiment: FormGroup;
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
    this.createExperiment = new FormGroup({
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
  }

  // Getters of experiment request form
  get experimentName() {
    return this.createExperiment.get('experimentName');
  }
  get description() {
    return this.createExperiment.get('description');
  }
  get frameworks() {
    return this.createExperiment.get('frameworks');
  }
  get namespace() {
    return this.createExperiment.get('namespace');
  }
  get cmd() {
    return this.createExperiment.get('cmd');
  }
  get envs() {
    return this.createExperiment.get('envs') as FormArray;
  }
  get image() {
    return this.createExperiment.get('image');
  }
  get specs() {
    return this.createExperiment.get('specs') as FormArray;
  }
  /**
   * Check the validity of the experiment page
   *
   */
  checkStatus() {
    if (this.current == 0) {
      return this.experimentName.invalid || this.description.invalid || this.frameworks.invalid || this.namespace.invalid || this.cmd.invalid;
    } else if (this.current == 1) {
      return this.image.invalid || this.envs.invalid;
    } else if (this.current == 2) {
      return this.specs.invalid;
    }
  }

  handleOk() {
    if (this.current === 1) {
      this.okText = 'Submit';
    }

    if (this.current < 2) {
      this.current ++;
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
        replicas: new FormControl(1, [Validators.min(1)]),
        cpus: new FormControl(1, [Validators.min(1)]),
        memory: new FormControl('', [this.experimentFormService.memoryValidator])
      },
      [
        this.experimentFormService.specValidator
      ]
    );
    this.specs.push(spec);
    // If the new page is created, jump to that page
    if (this.specs.controls.length > 1 && this.specs.controls.length % this.PAGESIZE === 1) {
      this.currentSpecPage += 1;
    }
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
    });
  }
  onDeleteExperiment(data: ExperimentInfo) {
    this.experimentService.deleteExperiment(data.experimentId).subscribe(
      () => {
        this.nzMessageService.success('Delete user success!');
        this.fetchExperimentList();
      },
      (err) => {
        this.nzMessageService.success(err.message);
      }
    );
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
