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

import { Component, Input, OnInit, ElementRef, ViewChild } from '@angular/core';
import { ModelService } from '@submarine/services/model.service';
import { ModelVersionService } from '@submarine/services/model-version.service';

@Component({
  selector: 'submarine-model-tags',
  templateUrl: './model-tags.component.html',
  styleUrls: ['./model-tags.component.scss'],
})
export class ModelTagsComponent implements OnInit {
  @Input() modelName: string = null;
  @Input() modelVersion: string = null;
  @Input() useVersionTags: boolean; // if true, use submarine model version tag, otherwise use submarine model tag.
  @Input() tags: Array<string>;
  @ViewChild('inputElement', { static: false }) inputElement?: ElementRef;
  inputVisible: boolean = false;
  inputValue: string = "";

  constructor(
    private modelVersionService: ModelVersionService,
    private modelService: ModelService,
  ) {}
  
  ngOnInit() {
     
  }

  deleteVersionTag = (tag: string) => {
    this.modelVersionService.deleteModelVersionTag(this.modelName, this.modelVersion, tag).subscribe(
      (res) => {
        this.tags = this.tags.filter(t => t !== tag);
      }
    );
  }

  deleteInfoTag = (tag: string) => {
    this.modelService.deleteModelTag(this.modelName, tag).subscribe(
      (res) => {
        this.tags = this.tags.filter(t => t !== tag);
      }
    );
  }

  showInput = () => {
    this.inputVisible = true;
    setTimeout(() => {
      this.inputElement.nativeElement.focus();
    }, 10);
  }

  handleInputConfirm = () => {
    if (this.inputValue) {
      const newTag = this.inputValue;
      if (!this.useVersionTags) {
        this.modelService.createModelTag(this.modelName, newTag).subscribe(
          (res) => {
            this.tags.push(newTag);
          }
        );
      }
      else {
        this.modelVersionService.createModelVersionTag(this.modelName, this.modelVersion, newTag).subscribe(
          (res) => {
            this.tags.push(newTag);
          }
        );
      }
    }
    this.inputValue = "";
    this.inputVisible = false;
  }
}
