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

import { Component, EventEmitter, Input, OnChanges, Output, SimpleChanges } from '@angular/core';
import { FormBuilder, FormControl, FormGroup, Validators } from '@angular/forms';
import { CascaderOption } from 'ng-zorro-antd/cascader';

interface DictItemInfo {
  code: string;
  name: string;
  status: string;
  edit: boolean;
}

@Component({
  selector: 'submarine-data-dict-config-modal',
  templateUrl: './data-dict-config-modal.component.html',
  styleUrls: ['./data-dict-config-modal.component.scss']
})
export class DataDictConfigModalComponent implements OnChanges {

  constructor(private fb: FormBuilder) {
    this.dictItemListForm = this.fb.group({
      dictItemCode: ['', [Validators.required]],
      dictItemName: ['', [Validators.required]]
    });

    this.newItemForm = this.fb.group({
      newItemCode: ['', [Validators.required]],
      newItemName: ['', [Validators.required]],
      newItemStatus: ['', [Validators.required]],
      selectedDictItemCode: ['', [Validators.required]],
      selectedDictItemName: ['', [Validators.required]],
      selectedDictItemStatus: ['', [Validators.required]]
    });
  }
  @Input() modalTitle: string; // Add | Edit
  @Input() dictCode: string;
  @Input() dictName: string;
  @Input() description: string;
  @Input() visible: boolean;
  @Input() modalWidth: number;
  @Input() lastDictCode: string;
  @Input() dictCodeChanged: boolean;
  @Input() newDictCode: string;
  @Output() readonly close: EventEmitter<any> = new EventEmitter();
  @Output() readonly ok: EventEmitter<any> = new EventEmitter();
  dictItemListForm: FormGroup;
  newItemForm: FormGroup;
  selectedItemIndex: number = 0;

  // TODO(kevin85421): mock data
  dictItemList: { [id: string]: DictItemInfo[] } = {
    "PROJECT_TYPE": [
      {
        code: 'PROJECT_TYPE_NOTEBOOK',
        name: 'notebook',
        status: 'available',
        edit: false
      },
      {
        code: 'PROJECT_TYPE_PYTHON',
        name: 'python',
        status: 'available',
        edit: false
      },
      {
        code: 'PROJECT_TYPE_R',
        name: 'R',
        status: 'available',
        edit: false
      },
      {
        code: 'PROJECT_TYPE_SCALA',
        name: 'scala',
        status: 'available',
        edit: false
      },
      {
        code: 'PROJECT_TYPE_TENSORFLOW',
        name: 'tensorflow',
        status: 'available',
        edit: false
      },
      {
        code: 'PROJECT_TYPE_PYTORCH',
        name: 'pytorch',
        status: 'available',
        edit: false
      }
    ],
    "PROJECT_VISIBILITY": [
      {
        code: 'PROJECT_VISIBILITY_PRIVATE',
        name: 'private',
        status: 'available',
        edit: false
      },
      {
        code: 'PROJECT_VISIBILITY_TEAM',
        name: 'team',
        status: 'available',
        edit: false
      },
      {
        code: 'PROJECT_VISIBILITY_PUBLIC',
        name: 'public',
        status: 'available',
        edit: false
      }
    ],
    "PROJECT_PERMISSION": [
      {
        code: 'PROJECT_PERMISSION_VIEW',
        name: 'can view',
        status: 'available',
        edit: false
      },
      {
        code: 'PROJECT_PERMISSION_EDIT',
        name: 'can edit',
        status: 'available',
        edit: false
      },
      {
        code: 'PROJECT_PERMISSION_EXECUTE',
        name: 'can execute',
        status: 'available',
        edit: false
      }
    ],
    "SYS_USER_SEX": [
      {
        code: 'SYS_USER_SEX_MALE',
        name: 'Male',
        status: 'available',
        edit: false
      },
      {
        code: 'SYS_USER_SEX_FEMALE',
        name: 'Female',
        status: 'available',
        edit: false
      }
    ],
    "SYS_USER_STATUS": [
      {
        code: 'SYS_USER_STATUS_AVAILABLE',
        name: 'Available',
        status: 'available',
        edit: false
      },
      {
        code: 'SYS_USER_STATUS_LOCKED',
        name: 'Locked',
        status: 'available',
        edit: false
      },
      {
        code: 'SYS_USER_STATUS_REGISTERED',
        name: 'New Registered',
        status: 'available',
        edit: false
      }
    ]
  };

  // Selected Item List
  selectedDictItemList: DictItemInfo[];

  // Add Item
  showNewRow: boolean = false;

  statusOptions: CascaderOption[] = [
    {
      value: 'available',
      label: 'available',
      isLeaf: true
    },
    {
      value: 'unavailable',
      label: 'unavailable',
      isLeaf: true
    }
  ];
  ngOnChanges(changes: SimpleChanges) {
    if (this.dictCodeChanged) {
      this.dictItemList[this.dictCode] = this.dictItemList[this.lastDictCode];
      this.dictCodeChanged = false;
      delete this.dictItemList[this.lastDictCode];
    }

    if (this.newDictCode !== '') {
      this.dictItemList[this.newDictCode] = [];
      this.newDictCode = '';
    }
    this.selectedDictItemList = this.dictItemList[this.dictCode];
  }

  hideModal() {
    this.cancelEdit(this.selectedItemIndex);
    this.cancelAddDictItem();
    this.close.emit();
  }

  // TODO(kevin85421)
  queryDictItem() {
    // Get query data
    for (const key in this.dictItemListForm.controls) {
      console.log(key);
      console.log(this.dictItemListForm.controls[key].value);
    }
  }

  startEdit(dictItemIndex: number, dictItemCode: string, dictItemName: string, dictItemStatus: string) {
    this.cancelAddDictItem();
    if (this.selectedItemIndex !== dictItemIndex) {
      this.cancelEdit(this.selectedItemIndex);
    }
    this.selectedDictItemList[dictItemIndex].edit = true;
    this.selectedItemIndex = dictItemIndex;
    this.newItemForm.setValue({ newItemCode: '', newItemName: '', newItemStatus: '',
                               selectedDictItemCode: dictItemCode, selectedDictItemName: dictItemName,
                               selectedDictItemStatus: dictItemStatus});
  }

  saveEdit(dictItemIndex: number) {
    this.selectedDictItemList[dictItemIndex].code = this.newItemForm.value.selectedDictItemCode;
    this.selectedDictItemList[dictItemIndex].name = this.newItemForm.value.selectedDictItemName;
    this.selectedDictItemList[dictItemIndex].status = this.newItemForm.value.selectedDictItemStatus;
    this.selectedDictItemList[dictItemIndex].edit = false;
  }

  cancelEdit(dictItemIndex: number) {
    this.selectedItemIndex = 0;
    this.newItemForm.setValue({ newItemCode: '', newItemName: '', newItemStatus: '',
                                selectedDictItemCode: '', selectedDictItemName: '',
                                selectedDictItemStatus: ''});
    if (this.selectedDictItemList.length !== 0) {
      this.selectedDictItemList[dictItemIndex].edit = false;
    }
  }

  // Add Item
  addDictItem() {
    this.cancelEdit(this.selectedItemIndex);
    this.showNewRow = true;
  }

  cancelAddDictItem() {
    this.showNewRow = false;
  }

  addNewItemToList() {
    this.selectedDictItemList = [
      {
        code: this.newItemForm.value.newItemCode,
        name: this.newItemForm.value.newItemName,
        status: this.newItemForm.value.newItemStatus,
        edit: false
      },
      ...this.selectedDictItemList
    ];
    this.dictItemList[this.dictCode] = [
      {
        code: this.newItemForm.value.newItemCode,
        name: this.newItemForm.value.newItemName,
        status: this.newItemForm.value.newItemStatus,
        edit: false
      },
      ...this.dictItemList[this.dictCode]
    ]
    this.newItemForm.setValue({ newItemCode: '', newItemName: '', newItemStatus: '',
                                selectedDictItemCode: '', selectedDictItemName: '',
                                selectedDictItemStatus: ''});
    this.cancelAddDictItem();
  }
}
