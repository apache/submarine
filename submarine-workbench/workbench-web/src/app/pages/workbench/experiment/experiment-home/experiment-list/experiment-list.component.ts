/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http: //www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { ExperimentInfo } from '@submarine/interfaces/experiment-info';

@Component({
  selector: 'submarine-experiment-list',
  templateUrl: './experiment-list.component.html',
  styleUrls: ['./experiment-list.component.scss'],
})
export class ExperimentListComponent implements OnInit {
  // property
  @Input() experimentList: ExperimentInfo[];
  @Input() isLoading: boolean;
  @Input() checkedList: boolean[];

  // event emitter
  @Output() deleteExperiment = new EventEmitter<number>();
  @Output() initModal = new EventEmitter<any>();

  // two-way binding: https://angular.io/guide/two-way-binding
  @Input() selectAllChecked: boolean;
  @Output() selectAllCheckedChange = new EventEmitter<boolean>();

  statusColor: { [key: string]: string } = {
    Accepted: 'gold',
    Created: 'white',
    Running: 'green',
    Succeeded: 'blue',
  };

  constructor() {}

  ngOnInit() {}

  onSelectAllClick() {
    for (let i = 0; i < this.checkedList.length; i++) {
      this.checkedList[i] = !this.selectAllChecked;
    }
    this.selectAllCheckedChange.emit(!this.selectAllChecked);
  }

  onDeleteExperiment(id: number) {
    this.deleteExperiment.emit(id);
  }
}
