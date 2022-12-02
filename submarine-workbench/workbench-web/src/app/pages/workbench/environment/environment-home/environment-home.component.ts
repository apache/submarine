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

import { Component, OnInit, ViewChild } from '@angular/core';
import { EnvironmentInfo } from '@submarine/interfaces/environment-interfaces/environment-info';
import { EnvironmentService } from '@submarine/services/environment-services/environment.service';
import { NzMessageService } from 'ng-zorro-antd';
import { EnvironmentFormComponent } from './environment-form/environment-form.component';
import { TranslateService } from '@ngx-translate/core';

@Component({
  selector: 'submarine-environment-home',
  templateUrl: './environment-home.component.html',
  styleUrls: ['./environment-home.component.scss'],
})
export class EnvironmentHomeComponent implements OnInit {
  environmentList: EnvironmentInfo[];

  @ViewChild('form', { static: true }) form: EnvironmentFormComponent;

  constructor(
    private environmentService: EnvironmentService,
    private nzMessageService: NzMessageService,
    private translate: TranslateService
    ) {
    }

  ngOnInit() {
    this.fetchEnvironmentList();
  }

  fetchEnvironmentList() {
    this.environmentService.fetchEnvironmentList().subscribe((res) => {
      this.environmentList = res;
    });
  }

  updateEnvironmentList(msg: string) {
    this.fetchEnvironmentList();
  }

  onDeleteEnvironment(name: string) {
    this.environmentService.deleteEnvironment(name).subscribe(
      () => {
        this.fetchEnvironmentList();
        this.nzMessageService.success(this.translate.instant('Delete') + ` ${name} ` + this.translate.instant('Success!'));
      },
      (err) => {
        this.nzMessageService.error(err);
      }
    );
  }
}
