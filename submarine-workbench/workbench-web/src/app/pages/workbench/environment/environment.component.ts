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
import { Environment } from '@submarine/interfaces/environment-info';
import { EnvironmentService } from '@submarine/services/environment.service';
import { NzMessageService } from 'ng-zorro-antd';

@Component({
  selector: 'submarine-environment',
  templateUrl: './environment.component.html',
  styleUrls: ['./environment.component.scss']
})
export class EnvironmentComponent implements OnInit {
  constructor(private environmentService: EnvironmentService, private nzMessageService: NzMessageService) {}

  environmentList: Environment[] = [];
  checkedList: boolean[] = [];
  selectAllChecked: boolean = false;

  isVisible = false;
  environmentForm;

  ngOnInit() {
    this.environmentForm = new FormGroup({
      environmentName: new FormControl(null, Validators.required),
      dockerImage: new FormControl(null, Validators.required),
      name: new FormControl(null, Validators.required),
      channels: new FormArray([]),
      dependencies: new FormArray([])
    });
    this.fetchEnvironmentList();
  }

  fetchEnvironmentList() {
    this.environmentService.fetchEnvironmentList().subscribe((list) => {
      this.environmentList = list;
      this.checkedList = [];
      for (let i = 0; i < this.environmentList.length; i++) {
        this.checkedList.push(false);
      }
    });
  }

  get environmentName() {
    return this.environmentForm.get('environmentName');
  }

  get dockerImage() {
    return this.environmentForm.get('dockerImage');
  }

  get name() {
    return this.environmentForm.get('name');
  }

  get channels() {
    return this.environmentForm.get('channels') as FormArray;
  }

  get dependencies() {
    return this.environmentForm.get('dependencies') as FormArray;
  }

  initEnvForm() {
    this.isVisible = true;
    this.environmentName.reset();
    this.dockerImage.reset();
    this.name.reset();
    this.channels.clear();
    this.dependencies.clear();
  }

  checkStatus() {
    return (
      this.environmentName.invalid ||
      this.dockerImage.invalid ||
      this.name.invalid ||
      this.channels.invalid ||
      this.dependencies.invalid
    );
  }

  closeModal() {
    this.isVisible = false;
  }

  addChannel() {
    this.channels.push(new FormControl('', Validators.required));
  }

  addDependencies() {
    this.dependencies.push(new FormControl('', Validators.required));
  }

  deleteItem(arr: FormArray, index: number) {
    arr.removeAt(index);
  }

  createEnvironment() {
    this.isVisible = false;
    const newEnvironmentSpec = this.createEnvironmentSpec();
    console.log(newEnvironmentSpec);
    this.environmentService.createEnvironment(newEnvironmentSpec).subscribe(
      () => {
        this.nzMessageService.success('Create Environment Success!');
        this.fetchEnvironmentList();
      },
      (err) => {
        this.nzMessageService.error(`${err}, please try again`, {
          nzPauseOnHover: true
        });
      }
    );
  }

  createEnvironmentSpec() {
    const environmentSpec = {
      name: this.environmentForm.get('environmentName').value,
      dockerImage: this.environmentForm.get('dockerImage').value,
      kernelSpec: {
        name: this.environmentForm.get('name').value,
        channels: [],
        dependencies: []
      }
    };

    for (const channel of this.channels.controls) {
      environmentSpec.kernelSpec.channels.push(channel.value);
    }

    for (const dependency of this.dependencies.controls) {
      environmentSpec.kernelSpec.dependencies.push(dependency.value);
    }

    return environmentSpec;
  }

  // TODO(kobe860219): Update an environment
  updateEnvironment(id: string, data) {}

  onDeleteEnvironment(name: string, onMessage: boolean) {
    this.environmentService.deleteEnvironment(name).subscribe(
      () => {
        if (onMessage === true) {
          this.nzMessageService.success('Delete Experiment Successfully!');
        }
        this.fetchEnvironmentList();
      },
      (err) => {
        if (onMessage === true) {
          this.nzMessageService.error(err.message);
        }
      }
    );
  }

  deleteEnvironments() {
    for (let i = this.checkedList.length - 1; i >= 0; i--) {
      console.log(this.environmentList[i].environmentSpec.name);
      if (this.checkedList[i] === true && this.environmentList[i].environmentSpec.name != 'notebook-env') {
        this.onDeleteEnvironment(this.environmentList[i].environmentSpec.name, false);
      }
    }

    this.selectAllChecked = false;
  }

  selectAllEnv() {
    for (let i = 0; i < this.checkedList.length; i++) {
      this.checkedList[i] = this.selectAllChecked;
    }
  }
}
