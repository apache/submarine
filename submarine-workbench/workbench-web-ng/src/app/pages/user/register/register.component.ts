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
import { FormBuilder, FormControl, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';

@Component({
  selector: 'submarine-register',
  templateUrl: './register.component.html',
  styleUrls: ['./register.component.scss']
})
export class RegisterComponent implements OnInit {
  validateForm: FormGroup;
  // TODO(kevin85421): the mock data must be removed in the future
  existedUsernames = ['test', 'haha'];
  existedEmails = ['test@gmail.com'];

  constructor(private fb: FormBuilder, private router: Router) {}
  ngOnInit(): void {
    this.validateForm = this.fb.group({
      username: [null, [Validators.required, this.checkExistedUsernames.bind(this)]],
      email: [null, [Validators.email, Validators.required, this.checkExistedEmails.bind(this)]],
      password: [null, [Validators.required, this.checkPasswordLength.bind(this)]],
      checkPassword: [null, [Validators.required, this.confirmationValidator.bind(this)]],
      agree: [false, this.agreeValidator.bind(this)]
    });
  }

  submitForm(): void {
    for (const i in this.validateForm.controls) {
      this.validateForm.controls[i].markAsDirty();
      this.validateForm.controls[i].updateValueAndValidity();
    }
    this.router.navigate(['/user/login']);
    console.log('SubmitForm (Sign up Page)');
  }

  updateConfirmValidator(): void {
    /** wait for refresh value */
    Promise.resolve().then(() => this.validateForm.get('checkPassword').updateValueAndValidity());
  }

  // Re-enter password must be the same with the password
  confirmationValidator = (control: FormControl): { [s: string]: boolean } => {
    if (!control.value) {
      return { required: true };
    } else if (control.value !== this.validateForm.get('password').value) {
      return { confirm: true, error: true };
    }
    return null;
  };

  // Agreement must be true
  agreeValidator = (control: FormControl): { [s: string]: boolean } => {
    if (control.value) {
      return null;
    } else {
      return { agree: false };
    }
  };

  // Username cannot be the same with existed usernames
  checkExistedUsernames(control: FormControl): { [s: string]: boolean } {
    if (this.existedUsernames.indexOf(control.value) !== -1) {
      return { usernameIsExisted: true };
    }
    return null;
  }

  // Email cannot be the same with existed emails
  checkExistedEmails(control: FormControl): { [s: string]: boolean } {
    if (this.existedEmails.indexOf(control.value) !== -1) {
      return { emailIsExisted: true };
    }
    return null;
  }

  // Password must be longer than 6 characters and shorter than 20 characters
  checkPasswordLength(control: FormControl): { [s: string]: boolean } {
    const controlValue = String(control.value);
    if (controlValue.length < 6 || controlValue.length > 20) {
      return { validPasswordLength: false };
    }
    return null;
  }
}
