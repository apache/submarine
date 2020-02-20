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

import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';

@Component({
  selector: 'app-project',
  templateUrl: './project.component.html',
  styleUrls: ['./project.component.scss']
})
export class ProjectComponent implements OnInit {
  newProject = false;
  existProjects = [];

  @ViewChild('inputElement', { static: false }) inputElement: ElementRef;

  constructor() { }

  //TODO(jasoonn): get projects data from server
  ngOnInit() {
    this.existProjects.push({
      projectName: 'projectName0', description: 'description', tags: ['12', 'Tag 2'], inputTagVisibility: false, projectInputTag: ''
    });
    this.existProjects.push({
      projectName: 'projectName1', description: 'description', tags: ['Unremovable', 'Tag 2'], inputTagVisibility: false, projectInputTag: ''
    });
    this.existProjects.push({
      projectName: 'projectName1', description: 'description', tags: ['Unremovable', 'Tag 2', 'Tag 3'], inputTagVisibility: false, projectInputTag: ''
    });
    this.existProjects.push({
      projectName: 'projectName1', description: 'description', tags: ['Unremovable', 'Tag 2', 'Tag 3'], inputTagVisibility: false, projectInputTag: ''
    })
  }
  //TODO(jasoonn): Update tag in server
  handleCloseTag(project, tag){
    project.tags = project.tags.filter(itag => itag!==tag);
    console.log(project);
    console.log(tag);
  }
  //TODO(jasoonn): update tag in server
  handleInputConfirm(project): void {
    if (project.projectInputTag && project.tags.indexOf(project.projectInputTag) === -1) {
      project.tags = [...project.tags, project.projectInputTag];
    }
    project.inputTagVisibility = false;
    project.projectInputTag = '';
  }

  showInput(project): void {
    project.inputTagVisibility = true;
    setTimeout(() => {
      this.inputElement.nativeElement.focus();
    }, 10);
  }
}
