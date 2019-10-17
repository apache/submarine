import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { NzNotificationService } from 'ng-zorro-antd';
import { AuthService } from '../../../services';

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
      this.router.navigate(['/manager/user']);
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
        user => {
          this.loginSuccess();
        },
        error => {
          console.log(error);
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
    this.router.navigate(['/manager/user']);

    setTimeout(() => {
      this.nzNotificationService.success('欢迎', '欢迎回来');
    }, 1000);
  }

  requestFailed(error: Error) {
    this.nzNotificationService.error('请求错误', error.message || '请求出现错误，请稍后再试', {
      nzDuration: 4000
    });
  }
}
