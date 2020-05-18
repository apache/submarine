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

import { Component, EventEmitter, Input, OnInit, Output, ViewChild } from '@angular/core';
import { NgForm } from '@angular/forms';
import { NzMessageService } from 'ng-zorro-antd';
import { UserService, ProjectService } from '@submarine/services';

interface AddProjectParams {
  name: string;
  userName: string;
  description: string;
  type: string;
  teamName: string;
  visibility: string;
  permission: string;
  starNum: number;
  likeNum: number;
  messageNum: number;
}

@Component({
  selector: 'submarine-new-project-page',
  templateUrl: './new-project-page.component.html',
  styleUrls: ['./new-project-page.component.scss']
})
export class NewProjectPageComponent implements OnInit {
  @Output() closeProjectPage = new EventEmitter<boolean>();
  @Output() addProject = new EventEmitter<AddProjectParams>();
  @ViewChild('f', { static: true }) signupForm: NgForm;
  // TODO(jasoonn): get team from API
  teams = ['ciil'];

  current = 0;
  initialState=0;

  templateType="Python";
  username = '';

  templateType = 'Python';

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

  constructor(
    private msg: NzMessageService,
    private projectService: ProjectService,
    private userService: UserService
  ) { }

  ngOnInit() {
    this.userService.fetchUserInfo().subscribe((data) => {
      this.username = data.username;
    })
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
      this.Templates.forEach(function (item, index, array) {
        if (item.type !== template.type) {
          array[index].checked = false;
        }
      });
      this.templateType = template.type;
    } else {
      this.templateType = '';
    }
  }
  
  done(): void{
    var project =  {
      name: this.newProjectContent.projectName,
      userName: this.username,
      description: this.newProjectContent.description,
      type: 'PROJECT_TYPE_NOTEBOOK',
      teamName: this.newProjectContent.team,
      visibility: this.newProjectContent.visibility,
      permission: this.newProjectContent.permission,
      starNum: 0,
      likeNum: 0,
      messageNum: 0
    }
    console.log(project)
    this.addProject.emit(project);
  }

  // TODO(jasoonn): open in notebook
  openNotebook() {}
}
