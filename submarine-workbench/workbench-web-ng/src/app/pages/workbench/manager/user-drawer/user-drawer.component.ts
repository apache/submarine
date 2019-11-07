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

import { Component, EventEmitter, Input, OnChanges, OnInit, Output, SimpleChanges } from '@angular/core';
import { FormBuilder, FormControl, FormGroup, ValidationErrors, Validators } from '@angular/forms';
import { SysUser } from '@submarine/interfaces';
import { SysDeptSelect } from '@submarine/interfaces/sys-dept-select';
import { SysDictItem } from '@submarine/interfaces/sys-dict-item';
import { SystemUtilsService, SysDictCode } from '@submarine/services';
import { format } from 'date-fns';
import { zip, Observable, Observer } from 'rxjs';
import { filter, map, startWith, take } from 'rxjs/operators';

@Component({
  selector: 'submarine-user-drawer',
  templateUrl: './user-drawer.component.html',
  styleUrls: ['./user-drawer.component.scss']
})
export class UserDrawerComponent implements OnInit, OnChanges {
  @Input() visible: boolean;
  @Input() readonly: boolean;
  @Input() sysUser: SysUser;
  @Input() sysDeptTreeList: SysDeptSelect[];
  @Output() readonly close: EventEmitter<any> = new EventEmitter<any>();
  @Output() readonly submit: EventEmitter<Partial<SysUser>> = new EventEmitter<Partial<SysUser>>();

  form: FormGroup;
  labelSpan = 5;
  controlSpan = 14;
  sexDictSelect: SysDictItem[] = [];
  statusDictSelect: SysDictItem[] = [];
  title = 'Add';

  constructor(private fb: FormBuilder, private systemUtilsService: SystemUtilsService) {
  }

  ngOnInit() {
    this.form = this.fb.group(
      {
        userName: new FormControl('',
          {
            updateOn: 'blur',
            validators: [Validators.required],
            asyncValidators: [this.userNameValidator]
          }
        ),
        password: ['', [Validators.required, Validators.pattern(/^(?=.*[a-zA-Z])(?=.*\d)(?=.*[~!@#$%^&*()_+`\-={}:";'<>?,./]).{8,}$/)]],
        confirm: ['', this.confirmValidator],
        realName: ['', Validators.required],
        deptCode: [],
        sex: [],
        status: [],
        birthday: [],
        email: new FormControl('', {
          updateOn: 'blur',
          validators: [Validators.email],
          asyncValidators: [this.emailValidator]
        }),
        phone: new FormControl('', {
          updateOn: 'blur',
          asyncValidators: [this.phoneValidator]
        })
      }
    );

    zip(
      this.systemUtilsService.fetchSysDictByCode(SysDictCode.USER_SEX),
      this.systemUtilsService.fetchSysDictByCode(SysDictCode.USER_STATUS)
    )
      .pipe(map(([item1, item2]) => [item1.records, item2.records]))
      .subscribe(([sexDictSelect, statusDictSelect]) => {
        this.sexDictSelect = sexDictSelect;
        this.statusDictSelect = statusDictSelect;
      });
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (!changes.visible) {
      return;
    }

    if (changes.visible.currentValue) {
      const sysUser = this.sysUser;
      const readOnly = this.readonly;

      this.form.reset({
        userName: sysUser ? sysUser.userName : '',
        password: sysUser ? sysUser.password : '',
        confirm: sysUser ? sysUser.password : '',
        realName: sysUser ? sysUser.realName : '',
        deptCode: {
          value: sysUser ? sysUser.deptCode : '',
          disabled: readOnly
        },
        sex: {
          value: sysUser ? sysUser.sex : '',
          disabled: readOnly
        },
        status: {
          value: sysUser ? sysUser.status : '',
          disabled: readOnly
        },
        birthday: {
          value: (sysUser && sysUser.birthday) ? new Date(sysUser.birthday) : '',
          disabled: readOnly
        },
        email: sysUser ? sysUser.email : '',
        phone: sysUser ? sysUser.phone : ''
      });
      this.title = this.generateTitle();
    }
  }

  onClose() {
    this.close.emit();
  }

  onSubmit() {
    for (const key in this.form.controls) {
      this.form.controls[key].markAsDirty();
      this.form.controls[key].updateValueAndValidity();
    }

    this.form.statusChanges.pipe(
      startWith(this.form.status),
      filter(status => status !== 'PENDING'),
      take(1),
      map(status => status === 'VALID')
    ).subscribe(valid => {
      if (valid) {
        const sysUser = this.sysUser ? this.sysUser : {};
        const formData = { ...sysUser, ...this.form.value };

        delete formData.confirm;
        delete formData['sex@dict'];
        delete formData['status@dict'];

        if (formData.birthday) {
          formData.birthday = format(formData.birthday, 'yyyy-MM-dd HH:mm:ss');
        }

        this.submit.emit(formData);
      }
    });
  }

  generateTitle(): string {
    if (this.readonly) {
      return 'Details';
    } else if (this.sysUser && this.sysUser.id) {
      return 'Edit';
    } else {
      return 'Add';
    }
  }

  confirmValidator = (control: FormControl): { [s: string]: boolean } => {
    if (!control.value) {
      return { error: true, required: true };
    } else if (control.value !== this.form.controls.password.value) {
      return { error: true, confirm: true };
    }

    return {};
  };

  validateConfirmPassword = () => {
    setTimeout(() => {
      this.form.controls.confirm.updateValueAndValidity();
    });
  };

  userNameValidator = (control: FormControl) =>
    new Observable((observer: Observer<ValidationErrors | null>) => {
      this.systemUtilsService
        .duplicateCheckUsername(control.value, this.sysUser && this.sysUser.id)
        .subscribe(status => {
          if (status) {
            observer.next(null);
          } else {
            observer.next({ error: true, duplicated: true });
          }

          observer.complete();
        });
    });

  emailValidator = (control: FormControl) =>
    new Observable((observer: Observer<ValidationErrors | null>) => {
      if (!control.value) {
        observer.next(null);
        return observer.complete();
      }

      this.systemUtilsService
        .duplicateCheckUserEmail(control.value, this.sysUser && this.sysUser.id)
        .subscribe(status => {
          if (status) {
            observer.next(null);
          } else {
            observer.next({ error: true, duplicated: true });
          }

          observer.complete();
        });
    });

  phoneValidator = (control: FormControl) =>
    new Observable((observer: Observer<ValidationErrors | null>) => {
      if (!control.value) {
        observer.next(null);
        return observer.complete();
      }

      this.systemUtilsService
        .duplicateCheckUserPhone(control.value, this.sysUser && this.sysUser.id)
        .subscribe(status => {
          if (status) {
            observer.next(null);
          } else {
            observer.next({ error: true, duplicated: true });
          }

          observer.complete();
        });
    });
}
