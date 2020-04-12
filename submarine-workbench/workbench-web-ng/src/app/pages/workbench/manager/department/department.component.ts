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
import { FormGroup, FormControl, Validators } from '@angular/forms';

export interface TreeNodeInterface {
  key: number;
  title: string; //need to be title so that can be displayed in ng tree node
  code: string;
  description: string;
  status: string;
  level?: number;
  expand?: boolean;
  children?: TreeNodeInterface[];
  parent?: TreeNodeInterface;
}

@Component({
  selector: 'app-department',
  templateUrl: './department.component.html',
  styleUrls: ['./department.component.scss']
})
export class DepartmentComponent implements OnInit {
  isLoading = true;
  dapartmentDictForm: FormGroup;
  newDepartmentForm: FormGroup;
  editMode = false;
  editNode: TreeNodeInterface;
  isVisible = false;
  isExpandTable = true;
  filterArr: TreeNodeInterface[] = [
    {
      key: 120,
      title: 'ABCD',
      code: '123',
      description: 'Company ABCD ',
      status: 'Deleted'
   }
  ];

  listOfMapData: TreeNodeInterface[] = [
    {
      key: 1,
      title: 'A',
      code: '123',
      description: 'Company A',
      status: 'Available',
      children: [
        {
          key: 11,
          title: 'AA',
          code: '123',
          description: 'Company AA',
          status: 'Deleted'
        },
        {
          key: 12,
          title: 'AB',
          code: '123',
          description: 'Company AB',
          status: 'Deleted',
          children: [
            {
              key: 121,
              title: 'ABC',
              code: '123',
              description: 'Company ABC',
              status: 'Deleted',
              children: [
                {
                  key: 120,
                  title: 'ABCD',
                  code: '123',
                  description: 'Company ABCD ',
                  status: 'Deleted'
                }
              ]
            }
          ]
        }
      ]
    },
    {
      key: 123,
      title: 'E',
      code: '999',
      description: 'Company E',
      status: 'Deleted'
    }
  ];

  constructor() { }

  //TODO(jasoonn): Load departments data
  ngOnInit() {
    this.dapartmentDictForm = new FormGroup({
      'departmentName': new FormControl(''),
      'departmentCode': new FormControl('')
    });
    this.newDepartmentForm = new FormGroup({
      'code': new FormControl(null, Validators.required),
      'name': new FormControl(null, Validators.required),
      'parent': new FormControl(null),
      'sort': new FormControl(0),
      'status': new FormControl(null),
      'description' : new FormControl(null)
    });
    setTimeout(() => {
      this.isLoading = false;
    }, 500);
    this.listOfMapData.forEach(item => {
      this.mapOfExpandedData[item.key] = this.convertTreeToList(item);
    });

    
  }

  queryDepartment(){
    this.filterArr = [];
    Object.keys(this.mapOfExpandedData).forEach(item => {
      this.mapOfExpandedData[item].forEach(node => {
        if (node.title.includes(this.dapartmentDictForm.get('departmentName').value) && node.code.includes(this.dapartmentDictForm.get('departmentCode').value)){
          console.log('bingo', node)
          this.filterArr.push(node);
        }
      });   
    });
    this.isExpandTable = false;
    this.filterArr=[...this.filterArr];
  }

  //TODO(jasoonn): Create or edit department, need to communicate with db
  submitDepartment(){
    //Edit department
    if (this.editMode === true){
      this.editNode.title = this.newDepartmentForm.get('name').value;
      this.editNode.code = this.newDepartmentForm.get('code').value;
      this.editNode.description = this.newDepartmentForm.get('description').value;
      this.editNode.status = this.newDepartmentForm.get('status').value ? 'Available' : 'Deleted';
    }
    else{
      console.log('createDepartment');
    }
    this.isVisible = false;
  }

  //TODO(jasoonn): Delete the department, need to comunicate with db
  deleteDepartment(node: TreeNodeInterface){
    node.status = 'Deleted';
  }

  //TODO(jasoonn): Restore the department, need to comunicate with db
  restoreDepartment(node: TreeNodeInterface){
    node.status = 'Available';
  }


  //TODO(jasoonn): Edit the department, need to comunicate with db, and reorder the list
  editDepartment(node: TreeNodeInterface){
    this.editNode = node;
    this.editMode = true;
    this.isVisible = true;
    this.newDepartmentForm.get('code').setValue(node.code);
    this.newDepartmentForm.get('name').setValue(node.title);
    if (node.parent){
      this.newDepartmentForm.get('parent').setValue(node.parent.key);
    }
    else {
      this.newDepartmentForm.get('parent').setValue(null);
    }
    this.newDepartmentForm.get('status').setValue(node.status === 'Available' ? true : false);
    this.newDepartmentForm.get('description').setValue(node.description);
  }

  mapOfExpandedData: { [key: string]: TreeNodeInterface[] } = {};

  collapse(array: TreeNodeInterface[], data: TreeNodeInterface, $event: boolean): void {
    if ($event === false) {
      if (data.children) {
        data.children.forEach(d => {
          const target = array.find(a => a.key === d.key)!;
          target.expand = false;
          if (target.key === data.key) return;
          this.collapse(array, target, false);
        });
      } else {
        return;
      }
    }
  }

  convertTreeToList(root: TreeNodeInterface): TreeNodeInterface[] {
    const stack: TreeNodeInterface[] = [];
    const array: TreeNodeInterface[] = [];
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

  visitNode(node: TreeNodeInterface, hashMap: { [key: string]: boolean }, array: TreeNodeInterface[]): void {
    if (!hashMap[node.key]) {
      hashMap[node.key] = true;
      array.push(node);
    }
  }

  showParent(node: TreeNodeInterface){
    if (node.parent) {
      return node.parent.title;
    }
    return 'None'
  }

}
