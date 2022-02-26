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

import { Component, ElementRef, Input, OnInit, ViewChild } from '@angular/core';

@Component({
    selector: 'register-model-tags',
    templateUrl: './register-model-tags.component.html',
    styleUrls: ['./register-model-tags.component.scss'],
})
export class RegisterModelTagsComponent implements OnInit {    
  @Input() tags: Array<string>
  @ViewChild('inputElement', { static: false }) inputElement?: ElementRef;
  inputVisible: boolean = false;
  inputValue: string = "";

  ngOnInit(): void {
  }

  handleInputConfirm = () => {
    if (this.inputValue) {
      const newTag = this.inputValue;
      if (!this.tags.includes(newTag)) this.tags.push(newTag);
    }
    this.inputValue = "";
    this.inputVisible = false;
  }

  showInput = () => {
    this.inputVisible = true;
    setTimeout(() => {
      this.inputElement.nativeElement.focus();
    }, 10);
  }

  deleteTag = (tag: string) => {
      this.tags = this.tags.filter(t => t !== tag);
  }

}