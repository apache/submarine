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

import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { ExperimentTemplate } from '@submarine/interfaces/experiment-template';
import { ExperimentFormService } from '@submarine/services/experiment.form.service';
import { ExperimentService } from '@submarine/services/experiment.service';
import { Subscription } from 'rxjs';
import { ExperimentTemplateSubmit } from '@submarine/interfaces/experiment-template-submit';
import { NzMessageService } from 'ng-zorro-antd';

interface ParsedTemplate {
  templateParams: {
    name: string;
    required: string;
    description: string;
    value: string;
  }[];
  experimentName: string;
  experimentNamespace: string;
  experimentCommand: string;
  experimentImage: string;
  experimentVars: string;
}

interface TemplateTable {
  [templateName: string]: ParsedTemplate;
}

@Component({
  selector: 'submarine-experiment-predefined-form',
  templateUrl: './experiment-predefined-form.component.html',
  styleUrls: ['./experiment-predefined-form.component.scss'],
})
export class ExperimentPredefinedFormComponent implements OnInit, OnDestroy {
  /* states that are bond to html template */
  paramList: { name: string; required: string }[];

  /* inner states */
  templates: TemplateTable = {};
  predefinedForm: FormGroup;
  subs: Subscription[] = [];

  constructor(
    private experimentService: ExperimentService,
    private experimentFormService: ExperimentFormService,
    private fb: FormBuilder,
    private nzMessageService: NzMessageService
  ) {}

  ngOnInit() {
    this.experimentService.fetchExperimentTemplateList().subscribe((res) => {
      this.templates = this.parseTemplateRespond(res);

      if (Object.keys(this.templates).length != 0) {
        // default: switch to first template
        const defaultTemplate = Object.keys(this.templates)[0];
        this.predefinedForm.get('templateName').setValue(defaultTemplate);
        this.onTemplateChange();
      }
    });

    this.predefinedForm = this.fb.group({
      templateName: [null],
      params: this.fb.group({}),
    });

    this.experimentFormService.modalPropsChange({
      okText: 'Submit',
    });

    this.predefinedForm.statusChanges.subscribe((status) => {
      this.experimentFormService.btnStatusChange(status === 'INVALID');
    });

    const sub = this.experimentFormService.stepService.subscribe((step) => {
      // handle submit
      this.onSubmit();
    });

    this.subs.push(sub);
  }
  ngOnDestroy() {
    this.subs.forEach((sub) => sub.unsubscribe());
  }
  onSubmit() {
    // construct spec
    const finalSpec: ExperimentTemplateSubmit = {
      params: this.predefinedForm.get('params').value,
    };
    const templateName = this.predefinedForm.get('templateName').value;

    // send http post request
    this.experimentService.createExperimentfromTemplate(finalSpec, templateName).subscribe({
      next: () => {},
      error: (msg) => {
        this.nzMessageService.error(`${msg}, please try again`, {
          nzPauseOnHover: true,
        });
      },
      complete: () => {
        this.nzMessageService.success('Experiment creation succeeds');
        this.experimentFormService.fetchList();
        this.experimentFormService.modalPropsClear();
        this.predefinedForm.reset();
      },
    });
  }
  parseTemplateRespond(res: ExperimentTemplate[]): TemplateTable {
    let templates: TemplateTable = {};
    for (let item of res) {
      // iterate template list
      let template: ParsedTemplate = {
        templateParams: item.experimentTemplateSpec.parameters.filter((item) => !item.name.startsWith('spec.')),
        experimentName: item.experimentTemplateSpec.experimentSpec.meta.name,
        experimentNamespace: item.experimentTemplateSpec.experimentSpec.meta.namespace,
        experimentCommand: item.experimentTemplateSpec.experimentSpec.meta.cmd,
        experimentImage: item.experimentTemplateSpec.experimentSpec.environment.image,
        experimentVars: JSON.stringify(item.experimentTemplateSpec.experimentSpec.meta.envVars),
      };
      templates[item.experimentTemplateSpec.name] = template;
    }
    return templates;
  }

  onTemplateChange() {
    if (this.currentOption == null) return;
    /* update paramList */
    let tmpList = [];
    for (let item of this.templates[this.currentOption].templateParams)
      tmpList.push({ name: item.name, required: item.required });
    this.paramList = tmpList;

    let controls = {};
    for (let item of this.templates[this.currentOption].templateParams) {
      controls[item.name] = [item.value];
      if (item.required === 'true') {
        if(item.name !== 'experiment_name') controls[item.name].push([Validators.required]);
        else controls[item.name].push([Validators.required, Validators.pattern('[a-zA-Z0-9][a-zA-Z0-9\-]*')]);
      }
    }
    const new_param_group = this.fb.group(controls);
    this.predefinedForm.setControl('params', new_param_group);
  }

  /* sugar syntax for option */
  get currentOption(): string {
    return this.predefinedForm.get('templateName').value;
  }
  get optionList(): string[] {
    return Object.keys(this.templates);
  }
}
