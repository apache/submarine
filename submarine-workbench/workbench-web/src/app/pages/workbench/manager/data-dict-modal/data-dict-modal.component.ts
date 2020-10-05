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

@Component({
  selector: 'submarine-data-dict-modal',
  templateUrl: './data-dict-modal.component.html',
  styleUrls: ['./data-dict-modal.component.scss']
})
export class DataDictModalComponent implements OnChanges {
  @Input() modalTitle: string; // Add | Edit
  @Input() dictCode: string;
  @Input() dictName: string;
  @Input() description: string;
  @Input() visible: boolean;
  @Output() readonly close: EventEmitter<any> = new EventEmitter();
  @Output() readonly ok: EventEmitter<any> = new EventEmitter();
  form: FormGroup;

  constructor(private fb: FormBuilder) {
    this.form = this.fb.group({
      dictCode: ['', Validators.required],
      dictName: ['', Validators.required],
      description: ['']
    });
  }

  ngOnChanges(changes: SimpleChanges) {
    this.form.reset({
      dictCode: this.dictCode,
      dictName: this.dictName,
      description: this.description
    });
  }

  hideModal() {
    this.close.emit();
  }

  submitForm() {
    for (const key in this.form.controls) {
      this.form.controls[key].markAsDirty();
      this.form.controls[key].updateValueAndValidity();
    }

    if (!this.form.valid) {
      return;
    }

    this.ok.emit(this.form.value);
  }
}
