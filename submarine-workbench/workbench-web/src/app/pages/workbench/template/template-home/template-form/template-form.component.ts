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

import { Component, OnInit, Output, EventEmitter } from '@angular/core';
import { FormArray, FormControl, FormGroup, Validators, FormBuilder } from '@angular/forms';
import { ExperimentService } from '@submarine/services/experiment.service';
import { ExperimentValidatorService } from '@submarine/services/experiment.validator.service';
import { Specs } from '@submarine/interfaces/experiment-spec';
import { ExperimentTemplateSpec } from '@submarine/interfaces/experiment-template';
import { NzMessageService } from 'ng-zorro-antd';
import { TranslateService } from '@ngx-translate/core';

@Component({
  selector: 'submarine-template-form',
  templateUrl: './template-form.component.html',
  styleUrls: ['./template-form.component.scss'],
})
export class TemplateFormComponent implements OnInit {
  @Output() private updater = new EventEmitter<string>();

  step: number = 0;

  defaultExperimentName = '{{experiment_name}}';

  isVisible: boolean;

  templateForm: FormGroup;
  finaleTemplate;

  jobType = 'Distributed Tensorflow';
  framework = 'Tensorflow';
  currentSpecPage = 1;
  PAGESIZE = 5;

  listOfOption: Array<{ label: string; value: string }> = [];

  // Constants
  TF_SPECNAMES = ['Master', 'Worker', 'Ps'];
  PYTORCH_SPECNAMES = ['Master', 'Worker'];
  defaultSpecName = 'worker';
  MEMORY_UNITS = ['M', 'G'];

  AUTHOR = 'admin';

  constructor(
    private experimentValidatorService: ExperimentValidatorService,
    private fb: FormBuilder,
    private experimentService: ExperimentService,
    private nzMessageService: NzMessageService,
    private translate: TranslateService
  ) {
  }

  ngOnInit() {
    //TODO: get tags from server
    this.listOfOption = []; 

    this.templateForm = this.fb.group({
      templateName: [null, Validators.required],
      description: [null, Validators.required],
      parameters: this.fb.array([], [this.experimentValidatorService.nameValidatorFactory('name')]),
      code: [null],
      specs: this.fb.array([], [this.experimentValidatorService.nameValidatorFactory('name')]),
      cmd: [null, Validators.required],
      tags: [[], Validators.required],
      envVars: this.fb.array([], [this.experimentValidatorService.nameValidatorFactory('key')]),
      image: [null, Validators.required],
    });
  }

  get templateName() {
    return this.templateForm.get('templateName');
  }

  get description() {
    return this.templateForm.get('description');
  }

  get parameters() {
    return this.templateForm.get('parameters') as FormArray;
  }

  get code() {
    return this.templateForm.get('code');
  }

  get specs() {
    return this.templateForm.get('specs') as FormArray;
  }

  get cmd() {
    return this.templateForm.get('cmd');
  }

  get tags() {
    return this.templateForm.get('tags');
  }

  get envVars() {
    return this.templateForm.get('envVars') as FormArray;
  }

  get image() {
    return this.templateForm.get('image');
  }

  initModal() {
    this.isVisible = true;
    this.initForm();
  }

  initForm() {
    this.templateName.reset();
    this.description.reset();
    this.parameters.clear();
    this.specs.clear();
    this.cmd.reset();
    this.tags.reset();
    this.envVars.clear();
    this.image.reset();
  }

  checkTemplateInfo() {
    return this.templateName.invalid || this.description.invalid || this.parameters.invalid;
  }

  checkExperimentInfo() {
    return this.image.invalid || this.cmd.invalid || this.tags.invalid || this.envVars.invalid;
  }

  checkResourceSpec() {
    return this.specs.invalid || this.specs.length < 1;
  }

  onCancel() {
    this.isVisible = false;
    this.step = 0;
  }

  createParam(defaultName: string = '', defaultValue: string = '') {
    return new FormGroup(
      {
        name: new FormControl(defaultName, [Validators.required]),
        value: new FormControl(defaultValue, [Validators.required]),
        required: new FormControl(true, [Validators.required]),
        description: new FormControl('', [Validators.required]),
      },
      [this.experimentValidatorService.paramValidator]
    );
  }

  onCreateParam() {
    const param = this.createParam();
    this.parameters.push(param);
  }

  createEnv(defaultKey: string = '', defaultValue: string = '') {
    return new FormGroup(
      {
        key: new FormControl(defaultKey, [Validators.required]),
        value: new FormControl(defaultValue, [Validators.required]),
      },
      [this.experimentValidatorService.envValidator]
    );
  }

  onCreateEnv() {
    const env = this.createEnv();
    this.envVars.push(env);
  }

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
            unit: new FormControl(defaultUnit, [Validators.required]),
          },
          [this.experimentValidatorService.memoryValidator]
        ),
      },
      [this.experimentValidatorService.specValidator]
    );
  }

  onCreateSpec() {
    const spec = this.createSpec();
    this.specs.push(spec);
  }

  deleteItem(arr: FormArray, index: number) {
    arr.removeAt(index);
  }

  deleteAllItem(arr: FormArray) {
    arr.clear();
  }

  createDefaultParameter() {
    return new FormGroup({
      name: new FormControl('experiment_name'),
      value: new FormControl(null),
      required: new FormControl(true),
      description: new FormControl('The name of experiment.'),
    });
  }

  constructTemplateSpec() {
    const defaultParameter = this.createDefaultParameter();
    this.parameters.push(defaultParameter);

    const specs: Specs = {};
    for (const spec of this.specs.controls) {
      if (spec.get('name').value) {
        specs[spec.get('name').value] = {
          replicas: spec.get('replicas').value,
          resources: `cpu=${spec.get('cpus').value},nvidia.com/gpu=${spec.get('gpus').value},memory=${
            spec.get('memory').get('num').value
          }${spec.get('memory').get('unit').value}`,
        };
      }
    }

    const envVars = {};
    for (const envVar of this.envVars.controls) {
      if (envVar.get('key').value) {
        envVars[envVar.get('key').value] = envVar.get('value').value;
      }
    }

    const newTemplateSpec: ExperimentTemplateSpec = {
      name: this.templateForm.get('templateName').value,
      author: this.AUTHOR,
      description: this.templateForm.get('description').value,
      parameters: this.templateForm.get('parameters').value,
      experimentSpec: {
        meta: {
          cmd: this.templateForm.get('cmd').value,
          name: this.defaultExperimentName,
          envVars: envVars,
          framework: this.framework,
          tags: this.templateForm.get('tags').value,
        },
        spec: specs,
        environment: {
          image: this.templateForm.get('image').value,
        },
      },
    };

    console.log(newTemplateSpec);
    return newTemplateSpec;
  }

  createTemplate() {
    const templateSpec = this.constructTemplateSpec();
    this.experimentService.createTemplate(templateSpec).subscribe({
      next: () => {},
      error: (msg) => {
        this.nzMessageService.error(`${msg}, ` + this.translate.instant('please try again'), {
          nzPauseOnHover: true,
        });
      },
      complete: () => {
        this.nzMessageService.success(this.translate.instant('Template creation succeeds'));
        this.isVisible = false;
        this.sendUpdate('Update Template List');
      },
    });
  }

  sendUpdate(updateInfo: string) {
    this.updater.emit(updateInfo);
  }
}
