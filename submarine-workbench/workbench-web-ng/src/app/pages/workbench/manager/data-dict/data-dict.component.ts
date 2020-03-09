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
import { FormBuilder, FormGroup } from '@angular/forms';
import { SysDictItem } from '@submarine/interfaces/sys-dict-item';

@Component({
  selector: 'submarine-data-dict',
  templateUrl: './data-dict.component.html',
  styleUrls: ['./data-dict.component.scss']
})

export class DataDictComponent implements OnInit {
  constructor(private fb: FormBuilder) {
  }
  dataDictForm: FormGroup;
  // TODO(kevin85421): (mock data) Replace it with sys-dict-item.ts
  sysDictList = [
    {
      dictCode: 'PROJECT_TYPE',
      dictName: 'Project machine learning type',
      description: 'submarine system dict, Do not modify.',
      status: 'available'
    },
    {
      dictCode: 'PROJECT_VISIBILITY',
      dictName: 'Project visibility type',
      description: 'submarine system dict, Do not modify.',
      status: 'available'
    },
    {
      dictCode: 'PROJECT_PERMISSION',
      dictName: 'Project permission type',
      description: 'submarine system dict, Do not modify.',
      status: 'available'
    },
    {
      dictCode: 'SYS_USER_SEX',
      dictName: 'Sys user sex',
      description: 'submarine system dict, Do not modify.',
      status: 'available'
    },
    {
      dictCode: 'SYS_USER_STATUS',
      dictName: 'Sys user status',
      description: 'submarine system dict, Do not modify.',
      status: 'available'
    }
  ];
  // modal
  modalTitle: string = '';
  // edit modal
  dataDictModalVisible: boolean = false;
  selectedDictCode: string = '';
  selectedDictName: string = '';
  selectedDescription: string = '';
  selectedDictID: number = 0;
  editDictCodeChanged: boolean = false;
  lastDictCode: string = '';

  // Configuration Modal
  configModalTitle: string = 'Dictionary Item List';
  configModalWidth: number = 1000;
  configModalVisible: boolean = false;

  // New Dict Item List
  newDictItemCode: string = '';

  ngOnInit() {
    this.dataDictForm = this.fb.group({
      dictName: [''],
      dictCode: ['']
    });
  }

  // TODO(kevin85421)
  queryDataDict() {}
  // TODO(kevin85421)
  onShowAddDataDictModal() {
    this.modalTitle = "Add";
    this.selectedDictCode = '';
    this.dataDictModalVisible = true;
  }

  // Edit Data Dictionary Modal
  onShowEditDataDictModal(data, dictItemIndex: number) {
    // set selected dict variables
    this.modalTitle = "Edit";
    this.selectedDictCode = data.dictCode;
    this.selectedDictName = data.dictName;
    this.selectedDescription = data.description;
    this.selectedDictID = dictItemIndex;
    this.lastDictCode = data.dictCode;
    this.editDictCodeChanged = false;
    // show edit modal
    this.dataDictModalVisible = true;
  }

  onHideDataDictModal() {
    // reset selected dict variables
    this.selectedDictName = '';
    this.selectedDescription = '';
    this.selectedDictID = 0;
    this.modalTitle = "";
    // hide edit modal
    this.dataDictModalVisible = false;
  }

  updateDataDict(dataDictItem: {dictCode: string, dictName: string, description: string}) {
    if (this.modalTitle === 'Edit') {
      if (this.sysDictList[this.selectedDictID].dictCode !== dataDictItem.dictCode) {
        this.editDictCodeChanged = true;
        this.sysDictList[this.selectedDictID].dictCode = dataDictItem.dictCode;
        this.selectedDictCode = dataDictItem.dictCode;
      }
      this.sysDictList[this.selectedDictID].dictName = dataDictItem.dictName;
      this.sysDictList[this.selectedDictID].description = dataDictItem.description;
    } else if (this.modalTitle === 'Add') {
      this.newDictItemCode = dataDictItem.dictCode;
      this.sysDictList = [
        ...this.sysDictList,
        {
          dictCode: dataDictItem.dictCode,
          dictName: dataDictItem.dictName,
          description: dataDictItem.description,
          status: 'available'
        }
      ]
    }

    this.onHideDataDictModal();
  }

  // Configuration
  onHideConfigModal() {
    this.selectedDictCode = '';
    this.selectedDictName = '';
    this.selectedDescription = '';
    this.selectedDictID = 0;
    this.configModalVisible = false;
  }

  onShowConfigModal(data, dictItemIndex: number) {
    this.selectedDictCode = data.dictCode;
    this.selectedDictName = data.dictName;
    this.selectedDescription = data.description;
    this.selectedDictID = dictItemIndex;
    this.configModalVisible = true;
  }

  // delete dataDictItem
  onDeleteDataDictItem(data) {
    this.sysDictList = this.sysDictList.filter(d => d.dictCode !== data.dictCode);
  }
}
