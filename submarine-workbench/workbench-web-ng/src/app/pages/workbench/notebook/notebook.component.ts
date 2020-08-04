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
import { NzMessageService } from 'ng-zorro-antd/message';
import { NzNotificationService } from 'ng-zorro-antd/notification';

@Component({
  selector: 'submarine-notebook',
  templateUrl: './notebook.component.html',
  styleUrls: ['./notebook.component.scss']
})
export class NotebookComponent implements OnInit {
  notebookList = [
    {
      status: 'Running',
      name: 'Notebook1',
      age: '35 mins',
      image: 'image1',
      cpu: '2',
      memory: '512 MB',
      volumes: 'volumes1'
    },
    {
      status: 'Stop',
      name: 'Notebook2',
      age: '40 mins',
      image: 'image2',
      cpu: '4',
      memory: '1024 MB',
      volumes: 'volumes2'
    }
  ];

  constructor(private message: NzMessageService, private notification: NzNotificationService) {}

  statusColor: { [key: string]: string } = {
    Running: 'green',
    Stop: 'blue'
  };

  ngOnInit() {
    this.message.warning('Notebook is in developing', { nzDuration: 5000 });
  }
}
