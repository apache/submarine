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
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthService } from '@submarine/services';
import { NzNotificationService } from 'ng-zorro-antd';

@Component({
  selector: 'submarine-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.scss']
})
export class LoginComponent implements OnInit {
  validateForm: FormGroup;

  constructor(
    private fb: FormBuilder,
    private authService: AuthService,
    private nzNotificationService: NzNotificationService,
    private router: Router
  ) {
    if (this.authService.isLoggedIn) {
      this.router.navigate(['/workbench']);
    }
  }

  submitForm(): void {
    for (const i in this.validateForm.controls) {
      this.validateForm.controls[i].markAsDirty();
      this.validateForm.controls[i].updateValueAndValidity();
    }

    if (this.validateForm.status === 'VALID') {
      const { value } = this.validateForm;
      this.authService.login(value).subscribe(
        () => {
          this.loginSuccess();
        },
        error => {
          this.requestFailed(error);
        }
      );
    }
  }

  ngOnInit(): void {
    this.validateForm = this.fb.group({
      userName: [null, [Validators.required]],
      password: [null, [Validators.required]],
      remember: [true]
    });
  }

  loginSuccess() {
    this.router.navigate(['/workbench']);
  }

  requestFailed(error: Error) {
    this.nzNotificationService.error('Request error', error.message || 'Request error, please try again', {
      nzDuration: 4000
    });
  }
}
