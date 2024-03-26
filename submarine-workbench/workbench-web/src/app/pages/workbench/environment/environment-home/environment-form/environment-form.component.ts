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

import { Component, EventEmitter, OnInit, Output } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { EnvironmentService } from '@submarine/services/environment-services/environment.service';
import { NzMessageService } from 'ng-zorro-antd';
import { parse } from 'yaml';
import { TranslateService } from '@ngx-translate/core';

@Component({
  selector: 'submarine-environment-form',
  templateUrl: './environment-form.component.html',
  styleUrls: ['./environment-form.component.scss'],
})
export class EnvironmentFormComponent implements OnInit {
  @Output() private updater = new EventEmitter<string>();

  isVisible: boolean;
  environmentForm: FormGroup;
  previewCondaConfig = '';

  constructor(
    private fb: FormBuilder,
    private environmentService: EnvironmentService,
    private nzMessageService: NzMessageService,
    private translate: TranslateService
  ) {
  }

  ngOnInit() {
    this.environmentForm = this.fb.group({
      environmentName: [null, Validators.required],
      dockerImage: [null, Validators.required],
    });
  }

  initModal() {
    this.isVisible = true;
    this.initFormStatus();
  }

  sendUpdate() {
    this.updater.emit('Update List');
  }

  get environmentName() {
    return this.environmentForm.get('environmentName');
  }

  get dockerImage() {
    return this.environmentForm.get('dockerImage');
  }

  initFormStatus() {
    this.isVisible = true;
    this.environmentName.reset();
    this.dockerImage.reset();
  }

  checkStatus() {
    return this.environmentName.invalid || this.dockerImage.invalid;
  }

  closeModal() {
    this.isVisible = false;
    this.previewCondaConfig = '';
  }

  beforeUpload = (file: File): boolean => {
    let reader = new FileReader();
    reader.readAsText(file);
    reader.onload = () => {
      this.previewCondaConfig = reader.result.toString();
      this.nzMessageService.success(`${file.name} ` + this.translate.instant('file read successfully.'));
      const config = parse(reader.result.toString());
    };
    return false;
  };

  checkCondaConfig(config): Object {
    if (config === null) {
      config = {};
    }
    if (!config['channels']) {
      config['channels'] = [];
    }
    if (!config['name']) {
      config['name'] = '';
    }
    config['condaDependencies'] = [];
    config['pipDependencies'] = [];
    return config;
  }

  parseCondaConfig(): Object {
    let config = this.checkCondaConfig(parse(this.previewCondaConfig));
    this.previewCondaConfig = '';
    try {
      if (config['dependencies'] !== undefined || null) {
        config['dependencies'].map((e: object | string) => {
          if (typeof e === 'object') {
            if (!e['pip']) {
              this.nzMessageService.error(this.translate.instant('dependencies include unknown object'));
              throw Error('dependencies include unknown object');
            } else {
              config['pipDependencies'] = e['pip'];
            }
          } else if (typeof e === 'string') {
            config['condaDependencies'].push(e);
          }
        });
      }
    } catch (error) {
      this.nzMessageService.error(this.translate.instant('Unable to parse the conda config file'));
      throw error;
    }
    return config;
  }

  createEnvironment() {
    this.isVisible = false;
    const newEnvironmentSpec = this.createEnvironmentSpec();
    this.environmentService.createEnvironment(newEnvironmentSpec).subscribe(
      () => {
        this.nzMessageService.success(this.translate.instant('Create Environment Success!'));
        this.sendUpdate();
      },
      (err) => {
        this.nzMessageService.error(`${err}, ` + this.translate.instant('please try again'), {
          nzPauseOnHover: true,
        });
      }
    );
  }

  createEnvironmentSpec() {
    let config = this.parseCondaConfig();
    const environmentSpec = {
      name: this.environmentForm.get('environmentName').value,
      dockerImage: this.environmentForm.get('dockerImage').value,
      kernelSpec: {
        name: config['name'],
        channels: config['channels'],
        condaDependencies: config['condaDependencies'],
        pipDependencies: config['pipDependencies'],
      },
    };
    return environmentSpec;
  }
}
