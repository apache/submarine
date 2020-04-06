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

import { Component, EventEmitter, OnInit, Output, ViewChild } from '@angular/core';
import { NgForm } from '@angular/forms';
import { NzMessageService } from 'ng-zorro-antd';

@Component({
  selector: 'submarine-new-project-page',
  templateUrl: './new-project-page.component.html',
  styleUrls: ['./new-project-page.component.scss']
})
export class NewProjectPageComponent implements OnInit {
  @Output() readonly closeProjectPage = new EventEmitter<boolean>();
  @ViewChild('f', { static: true }) signupForm: NgForm;

  // TODO(jasoonn): get team from API
  teams = ['ciil'];

  current = 0;
  initialState = 0;

  templateType = 'Python';
  isAllDisplayDataChecked = false;
  isIndeterminate = false;

  newProjectContent = {
    projectName: '',
    description: '',
    visibility: 'Private',
    team: '',
    permission: 'View',
    files: []
  };
  Templates = [
    { type: 'Python', description: 'Python Template', checked: true },
    { type: 'R', description: 'R Template', checked: false },
    { type: 'Spark', description: 'Spark Template', checked: false },
    { type: 'Tensorflow', description: 'Tensorflow Template', checked: false },
    { type: 'Pytorch', description: 'Pytorch Template', checked: false }
  ];

  constructor(private msg: NzMessageService) {
  }

  ngOnInit() {
  }

  handleChange({ file, fileList }): void {
    const status = file.status;
    if (status !== 'uploading') {
      console.log(file, fileList);
      console.log(this.newProjectContent.files);
    }
    if (status === 'done') {
      this.msg.success(`${file.name} file uploaded successfully.`);
      this.newProjectContent.files.push(file);
    } else if (status === 'error') {
      this.msg.error(`${file.name} file upload failed.`);
    }
  }

  clearProject() {
    this.closeProjectPage.emit(true);
  }

  refreshCheck(template) {
    if (template.checked === true) {
      this.Templates.forEach(function(item, index, array) {
        if (item.type !== template.type) {
          array[index].checked = false;
        }
      });
      this.templateType = template.type;
    } else {
      this.templateType = '';
    }
  }

  // TODO(jasoonn): Add the new project
  done(): void {
    console.log(this.newProjectContent);
    this.clearProject();
  }

  // TODO(jasoonn): open in notebook
  openNotebook() {
  }
}
