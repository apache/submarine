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
import { NotebookService } from '@submarine/services/notebook.service';

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

  // Namesapces
  namespacesList = [];
  currentNamespace;

  // Notebook list
  notebookList;
  notebookTable;

  constructor(private notebookService: NotebookService) {}

  statusColor: { [key: string]: string } = {
    Running: 'green',
    Stop: 'blue'
  };

  ngOnInit() {
    this.notebookForm = new FormGroup({
      notebookName: new FormControl(null, [Validators.required]),
      namespaces: new FormControl(this.currentNamespace, [Validators.required]),
      environment: new FormControl('env1', [Validators.required]),
      cpu: new FormControl(null, [Validators.required]),
      gpu: new FormControl(null, [Validators.required]),
      memory: new FormControl(null, [Validators.required])
    });

    this.fetchNotebookList();
  }

  fetchNotebookList() {
    this.notebookService.fetchNotebookList().subscribe((list) => {
      this.notebookList = list;
      this.checkedList = [];
      for (let i = 0; i < this.notebookList.length; i++) {
        this.checkedList.push(false);
      }
      // Get namespaces
      this.notebookList.forEach((element) => {
        if (this.namespacesList.indexOf(element.spec.meta.namespace) < 0) {
          this.namespacesList.push(element.spec.meta.namespace);
        }
      });
      // Set default namespace and table
      this.currentNamespace = this.namespacesList[0];
      this.notebookTable = [];
      this.notebookList.forEach((item) => {
        if (item.spec.meta.namespace == this.currentNamespace) {
          this.notebookTable.push(item);
        }
      });
    });
  }

  selectAllNotebook() {
    for (let i = 0; i < this.checkedList.length; i++) {
      this.checkedList[i] = this.checked;
    }
  }

  switchNamespace(namespace: string) {
    this.notebookTable = [];
    this.notebookList.forEach((item) => {
      if (item.spec.meta.namespace == namespace) {
        this.notebookTable.push(item);
      }
    });
    console.log(this.notebookTable);
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
