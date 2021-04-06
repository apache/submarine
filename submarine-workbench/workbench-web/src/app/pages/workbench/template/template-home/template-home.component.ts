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

import { Component, OnInit, ViewChild } from '@angular/core';
import { ExperimentTemplate } from '@submarine/interfaces/experiment-template';
import { ExperimentService } from '@submarine/services/experiment.service';
import { TemplateFormComponent } from './template-form/template-form.component';

@Component({
  selector: 'submarine-template-home',
  templateUrl: './template-home.component.html',
  styleUrls: ['./template-home.component.scss'],
})
export class TemplateHomeComponent implements OnInit {
  constructor(private experimentService: ExperimentService) {}

  templateList: ExperimentTemplate[];

  @ViewChild('form', { static: true }) form: TemplateFormComponent;

  ngOnInit() {
    this.fetchTemplateList();
  }

  fetchTemplateList() {
    this.experimentService.fetchExperimentTemplateList().subscribe((res) => {
      this.templateList = res;
      console.log(this.templateList);
    });
  }
}
