/*
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

import { Component, Input, OnDestroy, OnInit } from '@angular/core';
import { FormArray, FormControl, FormGroup, Validators } from '@angular/forms';
import { EnvironmentSpec, ExperimentMeta, ExperimentSpec, Specs } from '@submarine/interfaces/experiment-spec';
import { ExperimentFormService } from '@submarine/services/experiment.form.service';
import { ExperimentService } from '@submarine/services/experiment.service';
import { ExperimentValidatorService } from '@submarine/services/experiment.validator.service';
import { nanoid } from 'nanoid';
import { NzMessageService } from 'ng-zorro-antd';
import { Subscription } from 'rxjs';

@Component({
  selector: 'submarine-experiment-customized-form',
  templateUrl: './experiment-customized-form.component.html',
  styleUrls: ['./experiment-customized-form.component.scss']
})
export class ExperimentCustomizedFormComponent implements OnInit, OnDestroy {
  @Input() mode: 'create' | 'update' | 'clone';

  // About new experiment
  experiment: FormGroup;
  finialExperimentSpec: ExperimentSpec;
  step: number = 0;
  subscriptions: Subscription[] = [];

  // TODO: Fetch all namespaces from submarine server
  defaultNameSpace = 'default';
  nameSpaceList = [this.defaultNameSpace, 'submarine'];

  // TODO: Fetch all images from submarine server
  imageIndex = 0;
  defaultImage = 'gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0'
  imageList = [this.defaultImage];

  // Constants
  TF_SPECNAMES = ['Master', 'Worker', 'Ps'];
  PYTORCH_SPECNAMES = ['Master', 'Worker'];
  defaultSpecName = 'worker';
  MEMORY_UNITS = ['M', 'G'];

  SECOND_STEP = 1;
  PREVIEW_STEP = 2;
  ADVANCED = false;

  // About env page
  currentEnvPage = 1;
  PAGESIZE = 5;

  // About spec
  jobTypes = 'Distributed Tensorflow';
  framework = 'Tensorflow';
  currentSpecPage = 1;

  // About update
  @Input() targetId: string = null;
  @Input() targetSpec: ExperimentSpec = null;

  constructor(
    private experimentService: ExperimentService,
    private experimentValidatorService: ExperimentValidatorService,
    private experimentFormService: ExperimentFormService,
    private nzMessageService: NzMessageService
  ) {}

  ngOnInit() {
    this.experiment = new FormGroup({
      experimentName: new FormControl(null, Validators.required),
      description: new FormControl(null, [Validators.required]),
      namespace: new FormControl(this.defaultNameSpace, [Validators.required]),
      cmd: new FormControl('', [Validators.required]),
      image: new FormControl(this.defaultImage, [Validators.required]),
      envs: new FormArray([], [this.experimentValidatorService.nameValidatorFactory('key')]),
      specs: new FormArray([], [this.experimentValidatorService.nameValidatorFactory('name')])
    });

    // Bind the component method for callback
    this.checkStatus = this.checkStatus.bind(this);

    if (this.mode === 'update') {
      this.updateExperimentInit();
    } else if (this.mode === 'clone') {
      this.cloneExperimentInit(this.targetSpec);
    }

    // Fire status to parent when form value has changed
    const sub1 = this.experiment.valueChanges.subscribe(this.checkStatus);

    const sub2 = this.experimentFormService.stepService.subscribe((n) => {
      if (n > 0) {
        if (this.step === this.PREVIEW_STEP) {
          this.handleSubmit();
        } else if (this.step === this.SECOND_STEP) {
          this.onPreview();
          this.step += 1;
        } else {
          this.step += 1;
        }
      } else {
        this.step -= 1;
      }
      // Send the current step and okText back to parent
      this.experimentFormService.modalPropsChange({
        okText: this.step !== this.PREVIEW_STEP ? 'Next step' : 'Submit',
        currentStep: this.step
      });
      // Run check after step is changed
      this.checkStatus();
    });

    this.subscriptions.push(sub1, sub2);
  }

  ngOnDestroy() {
    // Clean up the subscriptions
    this.subscriptions.forEach((sub) => {
      sub.unsubscribe();
    });
  }

  addItem(input: HTMLInputElement): void {
    const value = input.value;
    if (this.imageList.indexOf(value) === -1) {
      this.imageList = [...this.imageList, input.value || `New item ${this.imageIndex++}`];
    }
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
   * Reset properties in parent component when the form is about to closed
   */
  closeModal() {
    this.experimentFormService.modalPropsClear();
  }

  /**
   * Check the validity of the experiment page
   */
  checkStatus() {
    if (this.step === 0) {
      this.experimentFormService.btnStatusChange(
        this.experimentName.invalid ||
          this.namespace.invalid ||
          this.cmd.invalid ||
          this.image.invalid ||
          this.envs.invalid
      );
    } else if (this.step === 1) {
      this.experimentFormService.btnStatusChange(this.specs.invalid);
    }
  }
  onPreview() {
    this.finialExperimentSpec = this.constructSpec();
  }
  /**
   * Event handler for Next step/Submit button
   */
  handleSubmit() {
    if (this.mode === 'create') {
      this.experimentService.createExperiment(this.finialExperimentSpec).subscribe({
        next: () => {},
        error: (msg) => {
          this.nzMessageService.error(`${msg}, please try again`, {
            nzPauseOnHover: true
          });
        },
        complete: () => {
          this.nzMessageService.success('Experiment creation succeeds');
          this.experimentFormService.fetchList();
          this.closeModal();
        }
      });
    } else if (this.mode === 'update') {
      this.experimentService.updateExperiment(this.targetId, this.finialExperimentSpec).subscribe(
        null,
        (msg) => {
          this.nzMessageService.error(`${msg}, please try again`, {
            nzPauseOnHover: true
          });
        },
        () => {
          this.nzMessageService.success('Modification succeeds!');
          this.experimentFormService.fetchList();
          this.closeModal();
        }
      );
    } else if (this.mode === 'clone') {
      this.experimentService.createExperiment(this.finialExperimentSpec).subscribe(
        null,
        (msg) => {
          this.nzMessageService.error(`${msg}, please try again`, {
            nzPauseOnHover: true
          });
        },
        () => {
          this.nzMessageService.success('Create a new experiment !');
          this.experimentFormService.fetchList();
          this.closeModal();
        }
      );
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
      [this.experimentValidatorService.envValidator]
    );
  }
  /**
   * Create a new spec
   */
  createSpec(
    defaultName: string = 'Worker',
    defaultReplica: number = 1,
    defaultCpu: number = 1,
    defaultGpu: number = 0,
    defaultMemory: number = 1024,
    defaultUnit: string = 'M'
  ): FormGroup {
    return new FormGroup(
      {
        name: new FormControl(defaultName, [Validators.required]),
        replicas: new FormControl(defaultReplica, [Validators.min(1), Validators.required]),
        cpus: new FormControl(defaultCpu, [Validators.min(1), Validators.required]),
        gpus: new FormControl(defaultGpu, [Validators.min(0), Validators.required]),
        memory: new FormGroup(
          {
            num: new FormControl(defaultMemory, [Validators.required]),
            unit: new FormControl(defaultUnit, [Validators.required])
          },
          [this.experimentValidatorService.memoryValidator]
        )
      },
      [this.experimentValidatorService.specValidator]
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
    const meta: ExperimentMeta = {
      name: this.experimentName.value,
      namespace: this.namespace.value,
      framework: this.framework === 'Standalone' ? 'Tensorflow' : this.framework,
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
          resources: `cpu=${spec.get('cpus').value},nvidia.com/gpu=${spec.get('gpus').value},memory=${
            spec.get('memory').get('num').value
          }${spec.get('memory').get('unit').value}`
        };
      }
    }

    const environment: EnvironmentSpec = {
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

  deleteAllItem(arr: FormArray) {
    arr.clear();
  }

  updateExperimentInit() {
    // Prevent user from modifying the name
    this.experimentName.disable();
    // Put value back
    this.experimentName.setValue(this.targetSpec.meta.name);
    this.cloneExperiment(this.targetSpec);
    // Check status to enable next btn
    this.checkStatus();
  }

  cloneExperimentInit(spec: ExperimentSpec) {
    // Enable user from modifying the name
    this.experimentName.enable();
    // Put value back
    const id: string = nanoid(8);
    const cloneExperimentName = spec.meta.name + '-' + id;
    this.experimentName.setValue(cloneExperimentName.toLocaleLowerCase());
    this.cloneExperiment(spec);
    this.checkStatus();
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
      const cpuCount = info.resourceMap.cpu;
      const gpuCount = info.resourceMap.gpu === undefined ? '0' : '1';
      const [memory, unit] = info.resourceMap.memory.match(/\d+|[MG]/g);
      const newSpec = this.createSpec(
        specName,
        parseInt(info.replicas, 10),
        parseInt(cpuCount, 10),
        parseInt(gpuCount, 10),
        parseInt(memory, 10),
        unit
      );
      this.specs.push(newSpec);
    }
  }
}
