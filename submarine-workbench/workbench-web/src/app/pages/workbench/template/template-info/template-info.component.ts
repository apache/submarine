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
import { ActivatedRoute, Router } from '@angular/router';
import { ExperimentTemplate } from '@submarine/interfaces/experiment-template';
import { ExperimentService } from '@submarine/services/experiment.service';
import { NzMessageService } from 'ng-zorro-antd';

@Component({
  selector: 'submarine-template-info',
  templateUrl: './template-info.component.html',
  styleUrls: ['./template-info.component.scss'],
})
export class TemplateInfoComponent implements OnInit {
  isLoading = true;
  templateName;
  templateInfo: ExperimentTemplate;
  templateVars: string;

  constructor(
    private router: Router,
    private route: ActivatedRoute,
    private experimentService: ExperimentService,
    private nzMessageService: NzMessageService
  ) {}

  ngOnInit() {
    this.templateName = this.route.snapshot.params.name;
    this.getTemplateInfo(this.templateName);
    this.experimentService.emitInfo(this.templateName);
  }

  getTemplateInfo(name: string) {
    this.experimentService.querySpecificTemplate(name).subscribe(
      (item) => {
        this.templateInfo = item;
        this.templateVars = JSON.stringify(this.templateInfo.experimentTemplateSpec.experimentSpec.meta.envVars);
        console.log(this.templateInfo.experimentTemplateSpec);
        this.isLoading = false;
      },
      (err) => {
        this.nzMessageService.error('Cannot load ' + name);
        this.router.navigate(['/workbench/template']);
      }
    );
  }

  deleteTemplate() {
    this.experimentService.deleteTemplate(this.templateName).subscribe(
      () => {
        this.router.navigate(['/workbench/template']);
      },
      (err) => {
        this.nzMessageService.error(err);
      }
    );
  }

  backHome() {
    this.router.navigate(['/workbench/template']);
    this.templateName = null;
    this.experimentService.emitInfo(this.templateName);
  }
}
