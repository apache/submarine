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
import { NzMessageService } from 'ng-zorro-antd/message';

@Component({
  selector: 'submarine-notebook',
  templateUrl: './notebook.component.html',
  styleUrls: ['./notebook.component.scss']
})
export class NotebookComponent implements OnInit {
  // Select Checked
  checkedList: boolean[] = [];
  selectAllChecked: boolean = false;

  // Namesapces
  allNamespaceList = [];
  currentNamespace;

  // Notebook list
  allNotebookList;
  notebookTable;

  constructor(private notebookService: NotebookService, private nzMessageService: NzMessageService) {}

  ngOnInit() {
    this.fetchNotebookList();
  }

  fetchNotebookList() {
    this.notebookService.fetchNotebookList().subscribe((list) => {
      this.allNotebookList = list;

      // Get namespaces
      this.getAllNamespaces();

      // Set default namespace and table
      this.setDefaultTable();

      this.checkedList = [];
      for (let i = 0; i < this.notebookTable.length; i++) {
        this.checkedList.push(false);
      }
    });
  }

  getAllNamespaces() {
    this.allNotebookList.forEach((element) => {
      if (this.allNamespaceList.indexOf(element.spec.meta.namespace) < 0) {
        this.allNamespaceList.push(element.spec.meta.namespace);
      }
    });
  }

  setDefaultTable() {
    this.currentNamespace = this.allNamespaceList[0];
    this.notebookTable = [];
    this.allNotebookList.forEach((item) => {
      if (item.spec.meta.namespace == this.currentNamespace) {
        this.notebookTable.push(item);
      }
    });
  }

  switchNamespace(namespace: string) {
    this.notebookTable = [];
    this.allNotebookList.forEach((item) => {
      if (item.spec.meta.namespace == namespace) {
        this.notebookTable.push(item);
      }
    });
    console.log(this.notebookTable);

    this.selectAllChecked = false;
    this.checkedList.length = 0;
    for (let i = 0; i < this.notebookTable.length; i++) {
      this.checkedList.push(false);
    }
  }

  selectAll() {
    for (let i = 0; i < this.checkedList.length; i++) {
      this.checkedList[i] = this.selectAllChecked;
    }
  }

  deleteNotebook(id: string, onMessage: boolean) {
    this.notebookService.deleteNotebook(id).subscribe(
      () => {
        if (onMessage === true) {
          this.nzMessageService.success('Delete Notebook Successfully!');
        }
      },
      (err) => {
        if (onMessage === true) {
          this.nzMessageService.success(err.message);
        }
      }
    );
  }

  deleteNotebooks() {
    for (let i = this.checkedList.length - 1; i >= 0; i--) {
      if (this.checkedList[i] === true) {
        this.deleteNotebook(this.notebookTable[i].notebookId, false);
      }
    }

    this.selectAllChecked = false;

    // Update NotebookTable
    this.updateNotebookTable();
  }

  updateNotebookTable() {
    this.notebookService.fetchNotebookList().subscribe((list) => {
      this.allNotebookList = list;
      this.notebookTable = [];
      this.allNotebookList.forEach((item) => {
        if (item.spec.meta.namespace == this.currentNamespace) {
          this.notebookTable.push(item);
        }
      });
      this.checkedList.length = 0;
      for (let i = 0; i < this.notebookTable.length; i++) {
        this.checkedList.push(false);
      }
    });
  }

  // TODO(kobe860219): Make a notebook run
  runNotebook() {}

  // TODO(kobe860219): Stop a running notebook
  stopNotebook() {}

  // TODO(kobe860219): Create new notebook
  createNotebook() {}
}
