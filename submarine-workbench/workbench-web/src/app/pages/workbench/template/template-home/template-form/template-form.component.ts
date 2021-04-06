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

import { Component, OnInit } from '@angular/core';
import { FormArray, FormControl, FormGroup, Validators } from '@angular/forms';
import { UserService } from '@submarine/services';
import { ExperimentValidatorService } from '@submarine/services/experiment.validator.service';
import { ExperimentTemplateParamSpec } from '@submarine/interfaces/experiment-template';

@Component({
  selector: 'submarine-template-form',
  templateUrl: './template-form.component.html',
  styleUrls: ['./template-form.component.scss'],
})
export class TemplateFormComponent implements OnInit {
  defaultNamespace = 'default';
  defaultExperimentName = '{{experiment_name}}';
  defaultParameters: ExperimentTemplateParamSpec = {
    name: 'experiment_name',
    value: null,
    required: 'true',
    description: 'The name of experiment.',
  };
  userId;

  isVisible: boolean;

  templateForm: FormGroup;
  experimentSpec: FormGroup;
  finaleTemplate;

  constructor(private experimentValidatorService: ExperimentValidatorService, private userService: UserService) {}

  ngOnInit() {
    this.userService.fetchUserInfo().subscribe((res) => {
      this.userId = res.id;
    });

    this.templateForm = new FormGroup({
      name: new FormControl(null, Validators.required),
      author: new FormControl(this.userId, Validators.required),
      description: new FormControl(null, Validators.required),
      parameters: new FormArray([], Validators.required),
    });

    this.experimentSpec = new FormGroup({
      code: new FormControl(null),
      specs: new FormArray([], [this.experimentValidatorService.nameValidatorFactory('name')]),
      meta: new FormGroup({
        experimentName: new FormControl(this.defaultExperimentName, Validators.required),
        framework: new FormControl(null, Validators.required),
        namespace: new FormControl(this.defaultNamespace, Validators.required),
        cmd: new FormControl(null, Validators.required),
        envs: new FormArray([], [this.experimentValidatorService.nameValidatorFactory('key')]),
      }),
      environment: new FormGroup({
        envName: new FormControl(null),
        envDescription: new FormControl(null),
        dockerImage: new FormControl(null),
        image: new FormControl(null, Validators.required),
        kernelSpec: new FormControl(null),
      }),
    });
  }

  initModal() {
    this.isVisible = true;
  }
}
