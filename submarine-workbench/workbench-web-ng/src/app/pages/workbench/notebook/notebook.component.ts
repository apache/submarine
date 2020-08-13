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
import { FormControl, FormGroup, Validators } from '@angular/forms';

@Component({
  selector: 'submarine-notebook',
  templateUrl: './notebook.component.html',
  styleUrls: ['./notebook.component.scss']
})
export class NotebookComponent implements OnInit {
  // Checkbox
  checkedList: boolean[] = [];
  checked: boolean = false;

  // New notebook modal
  cancelText = 'Cancel';
  okText = 'Create';
  isVisible = false;

  // New notebook(form)
  notebookForm: FormGroup;

  // Mock Data
  namespacesList = ['namespaces1', 'namespaces2'];
  currentNamespaces = this.namespacesList[0];
  notebookList = [
    {
      status: 'Running',
      name: 'Notebook1',
      age: '35 mins',
      environment: 'image1',
      cpu: '2',
      gpu: '1',
      memory: '512 MB',
      volumes: 'volumes1'
    },
    {
      status: 'Stop',
      name: 'Notebook2',
      age: '40 mins',
      environment: 'image2',
      cpu: '4',
      gpu: '4',
      memory: '1024 MB',
      volumes: 'volumes2'
    }
  ];

  constructor() {}

  statusColor: { [key: string]: string } = {
    Running: 'green',
    Stop: 'blue'
  };

  ngOnInit() {
    this.notebookForm = new FormGroup({
      notebookName: new FormControl(null, [Validators.required]),
      namespaces: new FormControl(this.currentNamespaces, [Validators.required]),
      environment: new FormControl('env1', [Validators.required]),
      cpu: new FormControl(null, [Validators.required]),
      gpu: new FormControl(null, [Validators.required]),
      memory: new FormControl(null, [Validators.required])
    });
    this.checkedList = [];
    for (let i = 0; i < this.notebookList.length; i++) {
      this.checkedList.push(false);
    }
  }

  selectAllNotebook() {
    for (let i = 0; i < this.checkedList.length; i++) {
      this.checkedList[i] = this.checked;
    }
  }

  // TODO(kobe860219): Make a notebook run
  runNotebook(data) {
    data.status = 'Running';
  }

  // TODO(kobe860219): Stop a running notebook
  stopNotebook(data) {
    data.status = 'Stop';
  }

  // TODO(kobe860219): Create new notebook
  createNotebook() {}

  // TODO(kobe860219): Delete notebook
  deleteNotebook() {}
}
