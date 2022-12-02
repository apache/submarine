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

import { Component, EventEmitter, OnInit, Output } from '@angular/core';
import { FormArray, FormControl, FormGroup, Validators, FormBuilder } from '@angular/forms';
import { ExperimentValidatorService } from '@submarine/services/experiment.validator.service';
import { EnvironmentService } from '@submarine/services/environment-services/environment.service';
import { NotebookService } from '@submarine/services/notebook-services/notebook.service';
import { UserService } from '@submarine/services/user.service';
import { NzMessageService } from 'ng-zorro-antd/message';
import { TranslateService } from '@ngx-translate/core';

@Component({
  selector: 'submarine-notebook-form',
  templateUrl: './notebook-form.component.html',
  styleUrls: ['./notebook-form.component.scss'],
})
export class NotebookFormComponent implements OnInit {
  isVisible: boolean;

  // User Information
  userId;

  // Environment
  envList;
  envNameList = [];
  indexOfDeaultEnv;

  // Form
  notebookForm: FormGroup;
  MEMORY_UNITS = ['M', 'Gi'];

  // refresh notebook list function
  @Output() private refreshNotebook = new EventEmitter<boolean>();

  constructor(
    private fb: FormBuilder,
    private experimentValidatorService: ExperimentValidatorService,
    private environmentService: EnvironmentService,
    private notebookService: NotebookService,
    private userService: UserService,
    private nzMessageService: NzMessageService,
    private translate: TranslateService
  ) {
  }

  ngOnInit() {
    this.userService.fetchUserInfo().subscribe((res) => {
      this.userId = res.id;
    });

    this.notebookForm = this.fb.group({
      notebookName: [
        null,
        [Validators.maxLength(63), Validators.pattern('^([a-z]|[a-z][-a-z0-9]*[a-z0-9])$'), Validators.required],
      ],
      envName: [null, Validators.required], // Environment
      envVars: this.fb.array([], [this.experimentValidatorService.nameValidatorFactory('key')]),
      cpus: [null, [Validators.min(0.1), Validators.required]],
      gpus: [null],
      memoryNum: [null, [Validators.required]],
      unit: [this.MEMORY_UNITS[0], [Validators.required]],
    });

    this.fetchEnvList();

    this.initFormStatus();
  }

  initModal() {
    this.isVisible = true;
    this.initFormStatus();
  }

  // Get all environment
  fetchEnvList() {
    this.environmentService.fetchEnvironmentList().subscribe((list) => {
      this.envList = list;
      this.envList.forEach((env) => {
        if (this.envNameList.indexOf(env.environmentSpec.name) < 0) {
          this.envNameList.push(env.environmentSpec.name);
        }
      });
      this.indexOfDeaultEnv = this.envNameList.indexOf('notebook-env');
    });
  }

  // Init Form
  initFormStatus() {
    this.notebookName.reset();
    this.envName.reset(this.envNameList[this.indexOfDeaultEnv]);
    this.envVars.clear();
    this.cpus.reset(1);
    this.gpus.reset(0);
    this.memoryNum.reset();
    this.unit.reset(this.MEMORY_UNITS[0]);
  }

  get notebookName() {
    return this.notebookForm.get('notebookName');
  }
  get envName() {
    return this.notebookForm.get('envName');
  }
  get envVars() {
    return this.notebookForm.get('envVars') as FormArray;
  }
  get cpus() {
    return this.notebookForm.get('cpus');
  }
  get gpus() {
    return this.notebookForm.get('gpus');
  }
  get memoryNum() {
    return this.notebookForm.get('memoryNum');
  }
  get unit() {
    return this.notebookForm.get('unit');
  }

  // EnvVars Form
  createEnvVar(defaultKey: string = '', defaultValue: string = '') {
    // Create a new FormGroup
    return new FormGroup(
      {
        key: new FormControl(defaultKey, [Validators.required]),
        value: new FormControl(defaultValue, [Validators.required]),
      },
      [this.experimentValidatorService.envValidator]
    );
  }

  // EnvVars Form
  onCreateEnvVar() {
    const env = this.createEnvVar();
    this.envVars.push(env);
  }

  // Delete item in EnvVars Form
  deleteItem(arr: FormArray, index: number) {
    arr.removeAt(index);
  }

  // Check form
  checkStatus() {
    return (
      this.notebookName.invalid ||
      this.envName.invalid ||
      this.cpus.invalid ||
      this.gpus.invalid ||
      this.memoryNum.invalid ||
      this.envVars.invalid
    );
  }

  // Develope submmit spec
  submitForm() {
    // Check GPU, then develope resources spec
    let resourceSpec;
    if (this.notebookForm.get('gpus').value === 0 || this.notebookForm.get('gpus').value == null) {
      resourceSpec = `cpu=${this.notebookForm.get('cpus').value},memory=${this.notebookForm.get('memoryNum').value}${
        this.notebookForm.get('unit').value
      }`;
    } else {
      resourceSpec = `cpu=${this.notebookForm.get('cpus').value},nvidia.com/gpu=${
        this.notebookForm.get('gpus').value
      },memory=${this.notebookForm.get('memoryNum').value}${this.notebookForm.get('unit').value}`;
    }

    // Develope submmit spec
    const newNotebookSpec = {
      meta: {
        name: this.notebookForm.get('notebookName').value,
        namespace: 'default',
        ownerId: this.userId,
      },
      environment: {
        name: this.notebookForm.get('envName').value,
      },
      spec: {
        envVars: {},
        resources: resourceSpec,
      },
    };

    for (const envVar of this.envVars.controls) {
      if (envVar.get('key').value) {
        newNotebookSpec.spec.envVars[envVar.get('key').value] = envVar.get('value').value;
      }
    }

    // Post
    this.notebookService.createNotebook(newNotebookSpec).subscribe({
      next: (result) => {},
      error: (msg) => {
        this.nzMessageService.error(`${msg}, ` + this.translate.instant('please try again'), {
          nzPauseOnHover: true,
        });
      },
      complete: () => {
        this.nzMessageService.info(this.translate.instant(`Notebook Creating`), {
          nzPauseOnHover: true,
        });
        this.isVisible = false;
        // refresh list after created a new notebook
        this.refreshNotebook.emit(true);
      },
    });
  }
}
