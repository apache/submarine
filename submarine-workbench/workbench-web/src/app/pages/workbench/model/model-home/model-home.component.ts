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
// import { ExperimentTemplate } from '@submarine/interfaces/experiment-template';
import { ExperimentService } from '@submarine/services/experiment.service';
// import { TemplateFormComponent } from './template-form/template-form.component';

@Component({
  selector: 'submarine-model-home',
  templateUrl: './model-home.component.html',
  styleUrls: ['./model-home.component.scss'],
})
export class ModelHomeComponent implements OnInit {
  constructor(private experimentService: ExperimentService) {}

  modelCards = [
    {
      'title': "Model One",
      'createTime': "2021-10-12",
      'updatedTime': "2021-10-13", 
      'tags': ["image", 'text'],
      'description': "first model",
    }, 
    {
      'title': "Model Two",
      'createTime': "2021-10-12",
      'updatedTime': "2021-10-13", 
      'tags': ["speech"],
      'description': "second model",
    },
  ];

//   @ViewChild('form', { static: true }) form: TemplateFormComponent;

  ngOnInit() {
    // this.fetchTemplateList();
  }

//   fetchTemplateList() {
//     this.experimentService.fetchExperimentTemplateList().subscribe((res) => {
//       this.templateList = res;
//     });
//   }

//   updateTemplateList(msg: string) {
//     this.fetchTemplateList();
//   }
}
