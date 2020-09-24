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
import { NotebookService } from '@submarine/services/notebook.service';
import { NzMessageService } from 'ng-zorro-antd/message';
import { EnvironmentService } from '@submarine/services/environment.service';
import { ExperimentValidatorService } from '@submarine/services/experiment.validator.service';

@Component({
  selector: 'submarine-notebook',
  templateUrl: './notebook.component.html',
  styleUrls: ['./notebook.component.scss']
})
export class NotebookComponent implements OnInit {
  // Environment
  envList;
  envNameList = [];

  // Namesapces
  allNamespaceList = [];
  currentNamespace;

  // Notebook list
  allNotebookList;
  notebookTable;

  // New Notebook Form
  notebookForm: FormGroup;
  isVisible = false;
  MEMORY_UNITS = ['M', 'Gi'];

  constructor(
    private notebookService: NotebookService,
    private nzMessageService: NzMessageService,
    private environmentService: EnvironmentService,
    private experimentValidatorService: ExperimentValidatorService
  ) {}

  ngOnInit() {
    this.notebookForm = new FormGroup({
      notebookName: new FormControl(null, Validators.required),
      envName: new FormControl(null, Validators.required), // Environment
      envVars: new FormArray([], [this.experimentValidatorService.nameValidatorFactory('key')]),
      cpus: new FormControl(null, [Validators.min(1), Validators.required]),
      gpus: new FormControl(null),
      memoryNum: new FormControl(null, [Validators.required]),
      unit: new FormControl(this.MEMORY_UNITS[0], [Validators.required])
    });
    this.fetchNotebookList();
    this.fetchEnvList();
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
    });
  }

  // Get all notebooks, then set default namespace.
  fetchNotebookList() {
    this.notebookService.fetchNotebookList().subscribe((list) => {
      this.allNotebookList = list;
    });
  }

  /* (Future work. If we need a api for get all namespaces.)
  getAllNamespaces() {
    this.allNotebookList.forEach((element) => {
      if (this.allNamespaceList.indexOf(element.spec.meta.namespace) < 0) {
        this.allNamespaceList.push(element.spec.meta.namespace);
      }
    });
  }
  */

  // Future work. If we have a api for get all namespaces.
  /*
  setDefaultTable() {
    this.currentNamespace = this.allNamespaceList[0];
    this.notebookTable = [];
    this.allNotebookList.forEach((item) => {
      if (item.spec.meta.namespace == this.currentNamespace) {
        this.notebookTable.push(item);
      }
    });
  }
  */

  // Future work. If we have a api for get all namespaces.
  switchNamespace(namespace: string) {
    this.notebookTable = [];
    this.allNotebookList.forEach((item) => {
      if (item.spec.meta.namespace == namespace) {
        this.notebookTable.push(item);
      }
    });
    console.log(this.notebookTable);
  }

  deleteNotebook(id: string) {
    this.notebookService.deleteNotebook(id).subscribe(
      () => {
        this.nzMessageService.success('Delete Notebook Successfully!');
        this.updateNotebookTable();
      },
      (err) => {
        this.nzMessageService.error(err.message);
      }
    );
  }

  // Create or Delete, then update Notebook Table
  updateNotebookTable() {
    this.notebookService.fetchNotebookList().subscribe((list) => {
      this.allNotebookList = list;
      this.notebookTable = [];
      this.allNotebookList.forEach((item) => {
        if (item.spec.meta.namespace == this.currentNamespace) {
          this.notebookTable.push(item);
        }
      });
    });
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

  // Init form when click create-btn
  initNotebookStatus() {
    this.isVisible = true;
    this.notebookName.reset();
    this.envName.reset(this.envNameList[0]);
    this.envVars.clear();
    this.cpus.reset(1);
    this.gpus.reset(0);
    this.memoryNum.reset();
    this.unit.reset(this.MEMORY_UNITS[0]);
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

  // Submmit
  handleOk() {
    this.createNotebookSpec();
  }

  // EnvVars Form
  createEnvVar(defaultKey: string = '', defaultValue: string = '') {
    // Create a new FormGroup
    return new FormGroup(
      {
        key: new FormControl(defaultKey, [Validators.required]),
        value: new FormControl(defaultValue, [Validators.required])
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

  // Develope submmit spec
  createNotebookSpec() {
    // Check GPU, then develope resources spec
    let resourceSpec;
    if (this.notebookForm.get('gpus').value === 0) {
      resourceSpec = `cpu=${this.notebookForm.get('cpus').value},memory=${this.notebookForm.get('memoryNum').value}${
        this.notebookForm.get('unit').value
      }`;
    } else {
      resourceSpec = `cpu=${this.notebookForm.get('cpus').value},gpu=${this.notebookForm.get('gpus').value},memory=${
        this.notebookForm.get('memoryNum').value
      }${this.notebookForm.get('unit').value}`;
    }

    // Develope submmit spec
    const newNotebookSpec = {
      meta: {
        name: this.notebookForm.get('notebookName').value,
        namespace: 'default'
      },
      environment: {
        name: this.notebookForm.get('envName').value
      },
      spec: {
        envVars: {},
        resources: resourceSpec
      }
    };

    for (const envVar of this.envVars.controls) {
      if (envVar.get('key').value) {
        newNotebookSpec.spec.envVars[envVar.get('key').value] = envVar.get('value').value;
      }
    }

    // console.log(newNotebookSpec);

    // Post
    this.notebookService.createNotebook(newNotebookSpec).subscribe({
      next: (result) => {
        this.fetchNotebookList();
      },
      error: (msg) => {
        this.nzMessageService.error(`${msg}, please try again`, {
          nzPauseOnHover: true
        });
      },
      complete: () => {
        this.nzMessageService.success('Notebook creation succeeds');
        this.isVisible = false;
      }
    });
  }

  // TODO(kobe860219): Make a notebook run
  runNotebook() {}

  // TODO(kobe860219): Stop a running notebook
  stopNotebook() {}
}
