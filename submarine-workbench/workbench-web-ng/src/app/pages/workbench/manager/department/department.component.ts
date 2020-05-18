/*!
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
import { FormControl, FormGroup, ValidationErrors, Validators } from '@angular/forms';
import { SysDeptItem } from '@submarine/interfaces/sys-dept-item';
import { SysDeptSelect } from '@submarine/interfaces/sys-dept-select';
import { DepartmentService } from '@submarine/services';
import { NzMessageService } from 'ng-zorro-antd';

@Component({
  selector: 'submarine-department',
  templateUrl: './department.component.html',
  styleUrls: ['./department.component.scss']
})
export class DepartmentComponent implements OnInit {
  // About display departments
  sysTreeParent: SysDeptSelect[] = [];
  sysDeptTreeList: SysDeptItem[] = [];
  listOfMapData: SysDeptItem[] = [];
  isLoading = true;
  dapartmentDictForm: FormGroup;
  isExpandTable = true;
  filterArr: SysDeptItem[] = [];
  mapOfExpandedData: { [key: string]: SysDeptItem[] } = {};
  // About new or edit department
  newDepartmentForm: FormGroup;
  editMode = false;
  editNode: SysDeptItem;
  submitBtnIsLoading = false;
  formCodeErrMsg = '';
  parentCodeValid = true;
  isVisible = false;

  constructor(private departmentService: DepartmentService, private nzMessageService: NzMessageService) {}

  ngOnInit() {
    this.dapartmentDictForm = new FormGroup({
      departmentName: new FormControl(''),
      departmentCode: new FormControl('')
    });
    this.newDepartmentForm = new FormGroup({
      code: new FormControl(
        null,
        [Validators.required, this.checkRequire.bind(this), this.checkIfParent.bind(this)],
        this.duplicateDeptCodeCheck.bind(this)
      ),
      name: new FormControl(null, Validators.required),
      parent: new FormControl(null, this.checkEditParent.bind(this)),
      sort: new FormControl(0),
      deleted: new FormControl(null),
      description: new FormControl('')
    });
    this.loadDepartment();
  }

  queryDepartment() {
    this.filterArr = [];
    Object.keys(this.mapOfExpandedData).forEach((item) => {
      this.mapOfExpandedData[item].forEach((node) => {
        if (
          node.deptName.includes(this.dapartmentDictForm.get('departmentName').value) &&
          node.deptCode.includes(this.dapartmentDictForm.get('departmentCode').value)
        ) {
          this.filterArr.push(node);
        }
      });
    });
    this.isExpandTable = false;
    this.filterArr = [...this.filterArr];
  }

  submitDepartment() {
    this.submitBtnIsLoading = true;
    if (this.editMode === true) {
      this.editNode.deptName = this.newDepartmentForm.get('name').value;
      this.editNode.deptCode = this.newDepartmentForm.get('code').value;
      this.editNode.description = this.newDepartmentForm.get('description').value;
      this.editNode.deleted = this.newDepartmentForm.get('deleted').value ? 0 : 1;
      this.editNode.sortOrder = this.newDepartmentForm.get('sort').value;
      this.editNode.parentCode = this.newDepartmentForm.get('parent').value;
      this.setDepartment(this.editNode);
    } else {
      this.departmentService
        .createDept({
          deptName: this.newDepartmentForm.get('name').value,
          deptCode: this.newDepartmentForm.get('code').value,
          sortOrder: this.newDepartmentForm.get('sort').value,
          description: this.newDepartmentForm.get('description').value,
          deleted: this.newDepartmentForm.get('deleted').value ? 0 : 1,
          parentCode: this.newDepartmentForm.get('parent').value
        })
        .subscribe(
          () => {
            this.nzMessageService.success('Add department success!');
            this.loadDepartment();
            this.isVisible = false;
            this.submitBtnIsLoading = false;
          },
          (err) => {
            this.nzMessageService.error(err.message);
            this.submitBtnIsLoading = false;
          }
        );
    }
  }

  addDept() {
    this.departmentService.fetchSysDeptSelect().subscribe((list) => {
      this.sysTreeParent = list;
    });
    this.parentCodeValid = true;
    this.newDepartmentForm = new FormGroup({
      code: new FormControl(
        '',
        [Validators.required, this.checkRequire.bind(this), this.checkIfParent.bind(this)],
        this.duplicateDeptCodeCheck.bind(this)
      ),
      name: new FormControl(null, Validators.required),
      parent: new FormControl(null),
      sort: new FormControl(0),
      deleted: new FormControl(null),
      description: new FormControl('')
    });
    this.formCodeErrMsg = 'Please entry department code!';
    this.isVisible = true;
    this.editMode = false;
  }

  loadDepartment() {
    this.departmentService.fetchSysDeptList().subscribe((list) => {
      this.listOfMapData = list;
      this.listOfMapData.forEach((item) => {
        this.mapOfExpandedData[item.key] = this.convertTreeToList(item);
      });
      this.isLoading = false;
    });
  }

  setDepartment(node: SysDeptItem) {
    this.departmentService
      .editDept({
        deptCode: node.deptCode,
        deptName: node.deptName,
        sortOrder: node.sortOrder,
        id: node.id,
        description: node.description,
        deleted: node.deleted,
        parentCode: node.parentCode
      })
      .subscribe(
        () => {
          this.nzMessageService.success('Update success!');
          this.loadDepartment();
          this.isVisible = false;
          this.submitBtnIsLoading = false;
        },
        (err) => {
          this.nzMessageService.error(err.message);
        }
      );
  }

  deleteDepartment(node: SysDeptItem) {
    node.deleted = 1;
    this.setDepartment(node);
  }

  restoreDepartment(node: SysDeptItem) {
    node.deleted = 0;
    this.setDepartment(node);
  }

  editDepartment(node: SysDeptItem) {
    this.departmentService.fetchSysDeptSelect().subscribe((list) => {
      this.sysTreeParent = list;
    });
    this.newDepartmentForm = new FormGroup({
      code: new FormControl(
        node.deptCode,
        [Validators.required, this.checkRequire.bind(this), this.checkIfParent.bind(this)],
        this.duplicateDeptCodeCheck.bind(this)
      ),
      name: new FormControl(node.deptName, Validators.required),
      parent: new FormControl(node.parent ? node.parent.key : null),
      sort: new FormControl(node.sortOrder),
      deleted: new FormControl(node.deleted === 0 ? true : false),
      description: new FormControl(node.description)
    });
    this.editNode = node;
    this.editMode = true;
    this.isVisible = true;
    this.parentCodeValid = true;
  }

  collapse(array: SysDeptItem[], data: SysDeptItem, $event: boolean): void {
    if ($event === false) {
      if (data.children) {
        data.children.forEach((d) => {
          const target = array.find((a) => a.key === d.key)!;
          target.expand = false;
          if (target.key === data.key) {
            return;
          }
          this.collapse(array, target, false);
        });
      } else {
        return;
      }
    }
  }

  convertTreeToList(root: SysDeptItem): SysDeptItem[] {
    const stack: SysDeptItem[] = [];
    const array: SysDeptItem[] = [];
    const hashMap = {};
    stack.push({ ...root, level: 0, expand: false });

    while (stack.length !== 0) {
      const node = stack.pop()!;
      this.visitNode(node, hashMap, array);
      if (node.children) {
        for (let i = node.children.length - 1; i >= 0; i--) {
          stack.push({ ...node.children[i], level: node.level! + 1, expand: false, parent: node });
        }
      }
    }
    return array;
  }

  visitNode(node: SysDeptItem, hashMap: { [key: string]: boolean }, array: SysDeptItem[]): void {
    if (!hashMap[node.key]) {
      hashMap[node.key] = true;
      array.push(node);
    }
  }

  showParent(node: SysDeptItem) {
    if (node.parent) {
      return node.parent.deptName;
    }
    return 'None';
  }

  duplicateDeptCodeCheck(control: FormControl): Promise<ValidationErrors | null> {
    const params = {
      tableName: 'sys_department',
      fieldName: 'dept_code',
      fieldVal: control.value,
      dataId: this.editMode ? this.editNode.id : undefined
    };
    const promise = new Promise((resolve, reject) => {
      this.departmentService.codeCheck(params).then(
        (success) => {
          if (success) {
            resolve(null);
          } else {
            this.formCodeErrMsg = 'This value already exists is not available!';
            resolve({ 'Duplicate Code': true });
          }
        },
        (err) => {
          reject(err);
        }
      );
    });
    return promise;
  }

  checkIfParent(control: FormControl): { [key: string]: any } | null {
    if (this.editMode) {
      if (this.editNode.children == null || this.editNode.deptCode === control.value) {
        return null;
      } else {
        console.log(this.newDepartmentForm.get('code'));
        const mesg = this.editNode.deptCode + 'is the parent code of other departments, can not be modified!';
        this.formCodeErrMsg = mesg;
        return { mesg: true };
      }
    } else {
      return null;
    }
  }

  checkRequire(control: FormControl): { [key: string]: any } | null {
    if (control.value === '') {
      const mesg = 'Please enter department code!';
      this.formCodeErrMsg = mesg;
      return { mesg: true };
    } else {
      this.formCodeErrMsg = '';
      return null;
    }
  }

  // Use code to find node
  checkNodeCode(node: SysDeptItem, targetCode: string, checkExistCode: string) {
    if (node.deptCode === targetCode) {
      this.checkCodeExist(node, checkExistCode);
    } else {
      if (node.children !== null) {
        for (let i = 0; i < node.children.length; i++) {
          this.checkNodeCode(node.children[i], targetCode, checkExistCode);
        }
      }
    }
  }

  // Check node exist under the node
  checkCodeExist(node: SysDeptItem, targetCode: string) {
    if (node.deptCode === targetCode) {
      this.parentCodeValid = false;
    } else {
      if (node.children !== null) {
        node.children.forEach((element) => {
          this.checkCodeExist(element, targetCode);
        });
      }
    }
  }

  checkEditParent(control: FormControl): { [key: string]: any } | null {
    this.parentCodeValid = true;
    if (this.editMode) {
      this.listOfMapData.forEach((element) => {
        this.checkNodeCode(element, this.editNode.deptCode, control.value);
      });
    }
    return null;
  }
}
