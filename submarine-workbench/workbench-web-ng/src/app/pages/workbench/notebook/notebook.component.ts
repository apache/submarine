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
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { NzMessageService } from 'ng-zorro-antd/message';

@Component({
  selector: 'submarine-notebook',
  templateUrl: './notebook.component.html',
  styleUrls: ['./notebook.component.scss']
})
export class NotebookComponent implements OnInit {
  isEditing = false;
  notebookList = [{ name: 'Notebook', createTime: '2020-05-16 20:00:00', createBy: 'Someone' }];

  //search
  notebookName: string = '';
  searchForm: FormGroup;

  //editor
  editorOptions = { theme: 'vs-dark', language: 'python' };
  code: string = `from tensorflow.examples.tutorials.mnist import input_data
  import numpy as np
  
  mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
  x_train = mnist.train.images
  y_train = mnist.train.labels
  x_test = mnist.test.images
  y_test = mnist.test.labels
  
  print(x_train.shape)
  print(y_train.shape)
  print(x_test.shape)
  print(y_test.shape)
  print("---")
  
  #print(x_train[1, :])
  print(np.argmax(y_train[1, :]))`;

  constructor(private fb: FormBuilder, private message: NzMessageService) {}

  ngOnInit() {
    this.message.warning('Notebook is in developing', { nzDuration: 5000 });

    this.searchForm = this.fb.group({
      notebookName: [this.notebookName]
    });
  }

  edit(notebook) {
    this.isEditing = true;
  }

  saveNotebook() {
    this.isEditing = false;
    this.message.success('Save notebook success!');
  }
}
